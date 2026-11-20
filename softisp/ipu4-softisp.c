// SPDX-License-Identifier: GPL-2.0
/*
 * Software ISP relay for the XPS 13 7390 2-in-1 front camera.
 *
 * Reads SBGGR10P (packed RAW10 Bayer) 1280x800 frames from the IPU4P
 * capture node at the sensor's native 60 fps, sums 2 frames per output
 * (3 or 4 in low light: 20 or 15 fps) and writes full-resolution 1280x800
 * YUYV to a v4l2loopback device.
 *
 * On demand: the sensor only streams while a reader streams from the
 * loopback (v4l2loopback client-usage event). When the last reader leaves,
 * the sensor stops after IDLE_MS and a black frame replaces the last image,
 * so a later reader never receives a stale picture.
 *
 * Pipeline: black level -> gray-world white balance -> Malvar-He-Cutler
 * demosaic -> sRGB gamma -> BT.601 limited-range YUYV. Auto exposure drives
 * sensor exposure lines first, then analogue gain.
 *
 * The sensor stays at its default vertical blanking: on this IPU4P, longer
 * frames make roughly a quarter of DMA transfers abort ~1 ms after start of
 * frame (firmware error HW_REPORTED_STR2MMIO), which shows up as stutter.
 *
 * Usage: ipu4-softisp <raw-capture-dev> <loopback-dev>
 * Environment: SOFTISP_DEBUG=1 prints statistics, SOFTISP_NO_AE=1 freezes
 * exposure, SOFTISP_VBLANK / SOFTISP_EXP_MAX override sensor timing (testing).
 */
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define W 1280
#define H 800
#define IN_STRIDE 1600		/* 1280 * 10 / 8 */
#define PAD 2			/* demosaic border */
#define PW (W + 2 * PAD)
#define PH (H + 2 * PAD)
#define N_BUFS 8			/* ~130 ms of slack at 60 fps */
#define BAD_FRAME_LIMIT 12
#define OVEREXPOSED_RUN 2	/* invalid frames in a row before lowering exposure */
#define MIN_ACC 2		/* sensor frames summed per output */
#define MAX_ACC 4
#define IDLE_MS 2000		/* keep streaming this long after last reader */
#define IDLE_REFRESH_MS 1000	/* rewrite the black frame this often while idle */
#define FRAME_TIMEOUT_MS 500	/* a stalled IPU stream; frames arrive every 17 ms */
#define FIRST_FRAME_TIMEOUT_MS 1500
#define MAX_RECOVERIES 3	/* consecutive failed sessions before exiting */
#define REOPEN_WAIT_MS 6000	/* the driver refuses opens until ISYS power cycles */
#define PAUSE_MARKER "/run/xps7390-webcam/paused"
#define PAUSE_MAX_MS 30000	/* resume on our own if the post-sleep hook never runs */

/*
 * System sleep: the systemd sleep hook sends SIGUSR1 before suspend and
 * waits for PAUSE_MARKER, so the sensor is stopped and the capture node
 * closed before the IPU is suspended (a stream that is live across suspend
 * dies and needs a reset). SIGUSR2 after resume restarts it for any readers.
 */
static volatile sig_atomic_t pause_req, resume_req;

static void on_sleep_signal(int sig)
{
	if (sig == SIGUSR1)
		pause_req = 1;
	else
		resume_req = 1;
}

#define BLACK_LEVEL 64		/* OV01A10 pedestal per frame, 10-bit */
#define LIN_MAX 4095		/* internal linear scale */
#define AE_TARGET 760.0		/* mean linear luma (~18% grey of LIN_MAX) */
#define AE_TOL 0.15		/* relative dead zone */
#define DEFAULT_VBLANK 96	/* sensor default: 60 fps, no DMA aborts */
#define GAIN_MIN 256		/* sensor analogue gain range */
#define GAIN_SWITCH 4096	/* above 16x analogue gain, sum more frames first */
#define DGAIN_MAX 4096		/* software gain limit, Q10 (4x) */
#define EXP_MIN 4		/* sensor exposure range, lines */

#define V4L2_EVENT_PRI_CLIENT_USAGE (V4L2_EVENT_PRIVATE_START + 0x08E00000 + 1)

static int exp_max = 888;	/* replaced by the sensor's limits at startup */
static int gain_max = 8192;
static uint16_t lin_lut[MAX_ACC * 1023 + 1];	/* frame sum -> linear */
static uint8_t gamma_lut[LIN_MAX + 1];			/* linear -> sRGB 8-bit */

static uint8_t raw_copy[IN_STRIDE * H];	/* cached copy of the DMA buffer */
static uint16_t acc[H][W];	/* sum of the frames in the current pair */
static int32_t pad[PH][PW];	/* linear, white-balanced, mirrored border */
static uint8_t out[H][W * 2];	/* YUYV */

static int xioctl(int fd, unsigned long req, void *arg)
{
	int r;

	do {
		r = ioctl(fd, req, arg);
	} while (r == -1 && errno == EINTR);
	return r;
}

/* find the sensor subdev that exposes analogue gain */
static int open_sensor_subdev(void)
{
	char path[64];
	int i, fd;

	for (i = 0; i < 32; i++) {
		struct v4l2_queryctrl qc = { .id = V4L2_CID_ANALOGUE_GAIN };

		snprintf(path, sizeof(path), "/dev/v4l-subdev%d", i);
		fd = open(path, O_RDWR);
		if (fd < 0)
			continue;
		if (xioctl(fd, VIDIOC_QUERYCTRL, &qc) == 0)
			return fd;
		close(fd);
	}
	return -1;
}

static int set_ctrl(int fd, unsigned int id, int value)
{
	struct v4l2_control c = { .id = id, .value = value };

	return xioctl(fd, VIDIOC_S_CTRL, &c);
}

/*
 * Sum of n frames -> linear. Scaled so that a 2-frame sum spans LIN_MAX:
 * summing more frames brightens the image like a longer exposure would.
 */
static void build_lin_lut(int n)
{
	const int black = n * BLACK_LEVEL;

	for (int i = 0; i <= MAX_ACC * 1023; i++) {
		int v = i <= black ? 0 :
			(int)((double)(i - black) * LIN_MAX /
			      (MIN_ACC * (1023 - BLACK_LEVEL)) + 0.5);
		lin_lut[i] = v > LIN_MAX ? LIN_MAX : v;
	}
}

static void build_gamma_lut(void)
{
	for (int i = 0; i <= LIN_MAX; i++) {
		double v = (double)i / LIN_MAX;

		v = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055;
		gamma_lut[i] = (uint8_t)(v * 255.0 + 0.5);
	}
}

static int env_int(const char *name, int fallback)
{
	const char *v = getenv(name);

	return v && *v ? atoi(v) : fallback;
}

/* A frame is invalid when it is mostly 0xff padding, or truncated: valid top
 * rows followed by padding, detected by an all-0xff final row. */
static int frame_invalid(const uint8_t *raw, int *truncated)
{
	const uint8_t *last = raw + (size_t)IN_STRIDE * (H - 1);
	unsigned int ff = 0, samples = 0, tail = 0;
	size_t off;

	for (off = 0; off < IN_STRIDE; off++)
		tail += last[off] == 0xff;
	if (tail == IN_STRIDE) {
		*truncated = 1;
		return 1;
	}
	for (off = 0; off < (size_t)IN_STRIDE * H; off += 32) {
		ff += raw[off] == 0xff;
		samples++;
	}
	return ff * 100 > samples * 95;
}

/* Unpack RAW10 (5 bytes -> 4 pixels) and add into the pair accumulator. */
static void accumulate(const uint8_t *raw, int first)
{
	for (int y = 0; y < H; y++) {
		const uint8_t *l = raw + (size_t)y * IN_STRIDE;
		uint16_t *o = acc[y];

		for (int x = 0; x < W; x += 4, l += 5) {
			uint16_t p0 = (l[0] << 2) | (l[4] & 3);
			uint16_t p1 = (l[1] << 2) | ((l[4] >> 2) & 3);
			uint16_t p2 = (l[2] << 2) | ((l[4] >> 4) & 3);
			uint16_t p3 = (l[3] << 2) | ((l[4] >> 6) & 3);

			if (first) {
				o[x] = p0; o[x + 1] = p1; o[x + 2] = p2; o[x + 3] = p3;
			} else {
				o[x] += p0; o[x + 1] += p1; o[x + 2] += p2; o[x + 3] += p3;
			}
		}
	}
}

struct stats {
	double r, g, b;		/* mean linear values */
	double meter;		/* highlight-capped, centre-weighted luma for AE */
	double wb_r, wb_g, wb_b;	/* sums over unclipped quads */
};

/* Bayer layout (SBGGR): even rows B Gb, odd rows Gr R. Subsampled stats. */
static void measure(struct stats *s)
{
	double n = 0, nc = 0, all = 0, centre = 0;

	memset(s, 0, sizeof(*s));
	for (int y = 0; y < H; y += 8) {
		for (int x = 0; x < W; x += 8) {
			int b = lin_lut[acc[y][x]];
			int g = (lin_lut[acc[y][x + 1]] + lin_lut[acc[y + 1][x]]) / 2;
			int r = lin_lut[acc[y + 1][x + 1]];

			/*
			 * Cap bright samples so a window or lamp cannot drag the
			 * exposure of the rest of the scene (and the face) down.
			 */
			double l = 0.30 * r + 0.55 * g + 0.15 * b;

			if (l > 2 * AE_TARGET)
				l = 2 * AE_TARGET;
			s->r += r; s->g += g; s->b += b;
			all += l;
			n++;
			if (x >= W / 4 && x < W * 3 / 4 && y >= H / 5 && y < H * 4 / 5) {
				centre += l;
				nc++;
			}
			if (r < LIN_MAX * 15 / 16 && g < LIN_MAX * 15 / 16 &&
			    b < LIN_MAX * 15 / 16) {
				s->wb_r += r; s->wb_g += g; s->wb_b += b;
			}
		}
	}
	s->r /= n; s->g /= n; s->b /= n;
	s->meter = 0.5 * (all / n) + 0.5 * (centre / nc);
}

/* Linearise, white-balance (Q10 gains) and fill the mirrored border. */
static void prepare(int gain_r_q10, int gain_g_q10, int gain_b_q10)
{
	for (int y = 0; y < H; y++) {
		int32_t *d = pad[y + PAD] + PAD;
		const uint16_t *s = acc[y];

		if (y & 1) {		/* Gr R */
			for (int x = 0; x < W; x += 2) {
				int g = (lin_lut[s[x]] * gain_g_q10) >> 10;
				int r = (lin_lut[s[x + 1]] * gain_r_q10) >> 10;

				d[x] = g > LIN_MAX ? LIN_MAX : g;
				d[x + 1] = r > LIN_MAX ? LIN_MAX : r;
			}
		} else {		/* B Gb */
			for (int x = 0; x < W; x += 2) {
				int b = (lin_lut[s[x]] * gain_b_q10) >> 10;
				int g = (lin_lut[s[x + 1]] * gain_g_q10) >> 10;

				d[x] = b > LIN_MAX ? LIN_MAX : b;
				d[x + 1] = g > LIN_MAX ? LIN_MAX : g;
			}
		}
		/* mirror columns, keeping Bayer parity: -1 -> 1, -2 -> 2 */
		d[-1] = d[1]; d[-2] = d[2];
		d[W] = d[W - 2]; d[W + 1] = d[W - 3];
	}
	for (int i = 1; i <= PAD; i++) {
		memcpy(pad[PAD - i], pad[PAD + i], sizeof(pad[0]));
		memcpy(pad[PAD + H - 1 + i], pad[PAD + H - 1 - i], sizeof(pad[0]));
	}
}

static inline int clamp_lin(int v)
{
	return v < 0 ? 0 : v > LIN_MAX ? LIN_MAX : v;
}

/*
 * Malvar-He-Cutler gradient-corrected bilinear demosaic, coefficients x16.
 * p points at the centre sample of the padded plane.
 */
#define P(dy, dx) p[(dy) * PW + (dx)]

static inline int g_at_rb(const int32_t *p)
{
	return clamp_lin((8 * P(0, 0) + 4 * (P(-1, 0) + P(1, 0) + P(0, -1) + P(0, 1)) -
			  2 * (P(-2, 0) + P(2, 0) + P(0, -2) + P(0, 2))) >> 4);
}

/* R at B, or B at R */
static inline int diag_at_rb(const int32_t *p)
{
	return clamp_lin((12 * P(0, 0) +
			  4 * (P(-1, -1) + P(-1, 1) + P(1, -1) + P(1, 1)) -
			  3 * (P(-2, 0) + P(2, 0) + P(0, -2) + P(0, 2))) >> 4);
}

/* colour whose samples sit left/right of this green */
static inline int horiz_at_g(const int32_t *p)
{
	return clamp_lin((10 * P(0, 0) + 8 * (P(0, -1) + P(0, 1)) -
			  2 * (P(0, -2) + P(0, 2)) -
			  2 * (P(-1, -1) + P(-1, 1) + P(1, -1) + P(1, 1)) +
			  (P(-2, 0) + P(2, 0))) >> 4);
}

/* colour whose samples sit above/below this green */
static inline int vert_at_g(const int32_t *p)
{
	return clamp_lin((10 * P(0, 0) + 8 * (P(-1, 0) + P(1, 0)) -
			  2 * (P(-2, 0) + P(2, 0)) -
			  2 * (P(-1, -1) + P(-1, 1) + P(1, -1) + P(1, 1)) +
			  (P(0, -2) + P(0, 2))) >> 4);
}

/* Two horizontally adjacent linear RGB pixels -> one YUYV macropixel. */
static inline void put_yuyv(uint8_t *d, int r0, int g0, int b0, int r1, int g1, int b1)
{
	int R0 = gamma_lut[r0], G0 = gamma_lut[g0], B0 = gamma_lut[b0];
	int R1 = gamma_lut[r1], G1 = gamma_lut[g1], B1 = gamma_lut[b1];
	int rs = R0 + R1, gs = G0 + G1, bs = B0 + B1;

	d[0] = ((66 * R0 + 129 * G0 + 25 * B0 + 128) >> 8) + 16;
	d[1] = ((-38 * rs - 74 * gs + 112 * bs + 256) >> 9) + 128;
	d[2] = ((66 * R1 + 129 * G1 + 25 * B1 + 128) >> 8) + 16;
	d[3] = ((112 * rs - 94 * gs - 18 * bs + 256) >> 9) + 128;
}

static void demosaic_to_yuyv(void)
{
	for (int y = 0; y < H; y += 2) {
		uint8_t *d0 = out[y], *d1 = out[y + 1];

		for (int x = 0; x < W; x += 2) {
			const int32_t *pb = &pad[y + PAD][x + PAD];	/* B  */
			const int32_t *pgb = pb + 1;			/* Gb */
			const int32_t *pgr = pb + PW;			/* Gr */
			const int32_t *pr = pb + PW + 1;		/* R  */

			put_yuyv(d0 + 2 * x,
				 diag_at_rb(pb), g_at_rb(pb), pb[0],
				 vert_at_g(pgb), pgb[0], horiz_at_g(pgb));
			put_yuyv(d1 + 2 * x,
				 horiz_at_g(pgr), pgr[0], vert_at_g(pgr),
				 pr[0], g_at_rb(pr), diag_at_rb(pr));
		}
	}
}

static void write_black(int out_fd)
{
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W * 2; x += 2) {
			out[y][x] = 16;		/* Y */
			out[y][x + 1] = 128;	/* U / V */
		}
	if (write(out_fd, out, sizeof(out)) < 0)
		perror("loopback write (black)");
}

/*
 * v4l2loopback's default frame interval is the unreduced 10000/300000, and
 * Chromium-based apps set it back to that when they capture. PipeWire 1.0
 * does not match it against 30/1, so GStreamer apps that request a frame
 * rate (Cheese) fail with "no more input formats". Advertise exactly 1/30.
 */
static void set_output_interval(int out_fd)
{
	struct v4l2_streamparm p = { .type = V4L2_BUF_TYPE_VIDEO_OUTPUT };

	p.parm.output.timeperframe.numerator = 1;
	p.parm.output.timeperframe.denominator = 30;
	if (xioctl(out_fd, VIDIOC_S_PARM, &p) < 0)
		perror("output S_PARM");
}

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/*
 * Drain pending v4l2loopback events; return the latest reader count. Call
 * only after poll() reported POLLPRI: on a blocking descriptor DQEVENT waits
 * for a new event when the queue is empty, so stop when none are pending.
 */
static int dq_readers(int out_fd, int current)
{
	struct v4l2_event ev;

	do {
		if (xioctl(out_fd, VIDIOC_DQEVENT, &ev) < 0)
			break;
		if (ev.type == V4L2_EVENT_PRI_CLIENT_USAGE)
			current = (int)*(const uint32_t *)ev.u.data;
	} while (ev.pending > 0);
	return current;
}

struct ae {
	int expo;	/* sensor exposure, lines */
	int gain;	/* sensor analogue gain, 256 = 1x */
	int nacc;	/* sensor frames summed per output */
	int dgain;	/* software gain, Q10 */
};

/*
 * Split the wanted exposure (lines x gain x frames x software gain) in order
 * of image quality: exposure lines, analogue gain up to GAIN_SWITCH, more
 * summed frames, the rest of the analogue gain, then software gain.
 */
static void ae_allocate(struct ae *a, double want)
{
	double e = want / ((double)GAIN_MIN * MIN_ACC), g, dg = 1.0;
	int n = a->nacc;

	if (e <= exp_max) {
		a->expo = e < EXP_MIN ? EXP_MIN : (int)e;
		a->gain = GAIN_MIN;
		a->nacc = MIN_ACC;
		a->dgain = 1024;
		return;
	}
	e = exp_max;
	g = want / (e * n);
	while (n < MAX_ACC && g > GAIN_SWITCH)
		g = want / (e * ++n);
	/* hysteresis: drop a frame only when well below the switch point */
	while (n > MIN_ACC && want / (e * (n - 1)) < GAIN_SWITCH * 0.8)
		g = want / (e * --n);
	if (g > gain_max) {
		dg = g / gain_max;
		g = gain_max;
	}
	if (g < GAIN_MIN)
		g = GAIN_MIN;
	if (dg * 1024 > DGAIN_MAX)
		dg = DGAIN_MAX / 1024.0;
	a->expo = exp_max;
	a->gain = (int)g;
	a->nacc = n;
	a->dgain = (int)(dg * 1024);
}

static void ae_apply(int sd_fd, const struct ae *a)
{
	if (sd_fd < 0)
		return;
	if (set_ctrl(sd_fd, V4L2_CID_EXPOSURE, a->expo) < 0)
		fprintf(stderr, "set exposure %d: %s\n", a->expo, strerror(errno));
	if (set_ctrl(sd_fd, V4L2_CID_ANALOGUE_GAIN, a->gain) < 0)
		fprintf(stderr, "set gain %d: %s\n", a->gain, strerror(errno));
}

/*
 * Halve the sensor exposure: analogue gain first, then exposure lines. The
 * frame sum and software gain do not affect clipping in the sensor, so they
 * return to their defaults. Used when frames arrive too clipped to meter.
 */
static void ae_darken(int sd_fd, struct ae *a)
{
	if (a->gain > GAIN_MIN)
		a->gain = a->gain / 2 > GAIN_MIN ? a->gain / 2 : GAIN_MIN;
	else
		a->expo = a->expo / 2 > EXP_MIN ? a->expo / 2 : EXP_MIN;
	a->dgain = 1024;
	if (a->nacc != MIN_ACC) {
		a->nacc = MIN_ACC;
		build_lin_lut(MIN_ACC);
	}
	ae_apply(sd_fd, a);
}

struct buf {
	void *start;
	size_t length;
};

struct isp {
	const char *in_name;
	int out_fd, sd_fd;
	struct ae ae;
	double wb_r, wb_b;
	unsigned long outputs, truncated, raw_frames;
	double rate_t0;		/* debug: start of the frame-rate window */
	unsigned long rate_raw0, rate_out0;
	int readers;		/* last reader count from v4l2loopback */
};

/*
 * Stream the sensor until no reader has been active for IDLE_MS.
 * Returns 0 after a clean stop, 1 when stopped for system sleep, -1 on a
 * capture failure (stall or invalid frames). The loopback output stays open
 * either way, so readers keep their streams across a recovery or a sleep.
 */
static int stream_session(struct isp *s)
{
	struct buf bufs[N_BUFS];
	unsigned int nbufs, i, bad_frames = 0;
	int in_fd = -1, phase = 0, readers = s->readers > 0 ? s->readers : 1, ret = -1;
	unsigned long session_frames = 0;
	double idle_since = 0, waited = 0, last_frame;

	/* After a firmware hang the driver returns EIO until ISYS power cycles. */
	while ((in_fd = open(s->in_name, O_RDWR)) < 0 && errno == EIO &&
	       waited < REOPEN_WAIT_MS && !pause_req) {
		usleep(250 * 1000);
		waited += 250;
	}
	if (in_fd < 0 && pause_req)
		return 1;
	if (in_fd < 0) {
		perror(s->in_name);
		return -1;
	}
	{
		struct v4l2_format f = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };

		f.fmt.pix.width = W;
		f.fmt.pix.height = H;
		f.fmt.pix.pixelformat = v4l2_fourcc('p', 'B', 'A', 'A');
		f.fmt.pix.field = V4L2_FIELD_NONE;
		if (xioctl(in_fd, VIDIOC_S_FMT, &f) < 0) {
			perror("input S_FMT");
			close(in_fd);
			return -1;
		}
	}
	{
		struct v4l2_requestbuffers rb = {
			.count = N_BUFS,
			.type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
			.memory = V4L2_MEMORY_MMAP,
		};

		if (xioctl(in_fd, VIDIOC_REQBUFS, &rb) < 0) {
			perror("REQBUFS");
			close(in_fd);
			return -1;
		}
		nbufs = rb.count < N_BUFS ? rb.count : N_BUFS;
	}
	for (i = 0; i < nbufs; i++)
		bufs[i].start = MAP_FAILED;
	for (i = 0; i < nbufs; i++) {
		struct v4l2_buffer b = {
			.index = i,
			.type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
			.memory = V4L2_MEMORY_MMAP,
		};

		if (xioctl(in_fd, VIDIOC_QUERYBUF, &b) < 0) {
			perror("QUERYBUF");
			goto out;
		}
		bufs[i].length = b.length;
		bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE,
				     MAP_SHARED, in_fd, b.m.offset);
		if (bufs[i].start == MAP_FAILED) {
			perror("mmap");
			goto out;
		}
		if (xioctl(in_fd, VIDIOC_QBUF, &b) < 0) {
			perror("QBUF");
			goto out;
		}
	}
	ae_apply(s->sd_fd, &s->ae);
	{
		int t = V4L2_BUF_TYPE_VIDEO_CAPTURE;

		if (xioctl(in_fd, VIDIOC_STREAMON, &t) < 0) {
			perror("STREAMON");
			goto out;
		}
	}
	last_frame = now_ms();

	for (;;) {
		struct v4l2_buffer b = {
			.type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
			.memory = V4L2_MEMORY_MMAP,
		};
		struct pollfd pfd[2] = {
			{ .fd = in_fd, .events = POLLIN },
			{ .fd = s->out_fd, .events = POLLPRI },
		};
		struct stats st;
		int cut = 0;

		if (pause_req) {
			ret = 1;
			goto out;
		}
		int pr = poll(pfd, 2, session_frames ? FRAME_TIMEOUT_MS
						     : FIRST_FRAME_TIMEOUT_MS);

		if (pr < 0 && errno == EINTR)
			continue;	/* a sleep signal; checked above */
		if (pr <= 0) {
			fprintf(stderr, "ipu4-softisp: capture stalled after %lu frames\n",
				session_frames);
			goto out;
		}
		if (pfd[1].revents & POLLPRI) {
			readers = dq_readers(s->out_fd, readers);
			s->readers = readers;
			idle_since = readers > 0 ? 0 : idle_since ? idle_since : now_ms();
		}
		if (idle_since && now_ms() - idle_since > IDLE_MS) {
			ret = 0;
			goto out;
		}
		if (!(pfd[0].revents & POLLIN)) {
			if (now_ms() - last_frame > (session_frames ? FRAME_TIMEOUT_MS
								  : FIRST_FRAME_TIMEOUT_MS)) {
				fprintf(stderr, "ipu4-softisp: capture stalled after %lu frames\n",
					session_frames);
				goto out;
			}
			continue;
		}
		last_frame = now_ms();
		if (xioctl(in_fd, VIDIOC_DQBUF, &b) < 0) {
			perror("DQBUF");
			goto out;
		}
		/*
		 * The DMA buffers are slow to read in place, so copy each frame
		 * once with wide sequential loads and return the buffer at once.
		 */
		memcpy(raw_copy, bufs[b.index].start, sizeof(raw_copy));
		if (xioctl(in_fd, VIDIOC_QBUF, &b) < 0) {
			perror("re-QBUF");
			goto out;
		}

		/*
		 * A failed CSI session returns a short valid prefix followed by
		 * 0xff padding while still reporting a full buffer. Never pass
		 * those to applications; persistent failures restart the stream.
		 */
		if (frame_invalid(raw_copy, &cut)) {
			s->truncated += cut;
			phase = 0;	/* drop the incomplete sum */
			/*
			 * A fully clipped row packs to the same 0xff bytes as DMA
			 * padding, so a scene far brighter than the current exposure
			 * (daylight after a dark session, a bright light coming into
			 * view) lands here, where auto exposure never sees it. After a
			 * run of invalid frames, halve the sensor exposure on each one;
			 * if the cause was a real truncation, auto exposure restores
			 * it within a few frames. The reduction carries into a restart.
			 */
			if (++bad_frames >= OVEREXPOSED_RUN && !getenv("SOFTISP_NO_AE")) {
				if (bad_frames == OVEREXPOSED_RUN)
					fprintf(stderr,
						"ipu4-softisp: invalid frames, possibly overexposed; lowering exposure\n");
				ae_darken(s->sd_fd, &s->ae);
			}
			if (bad_frames >= BAD_FRAME_LIMIT) {
				fprintf(stderr,
					"ipu4-softisp: invalid padded frames; restarting stream\n");
				goto out;
			}
			continue;
		}
		bad_frames = 0;
		s->raw_frames++;
		session_frames++;

		accumulate(raw_copy, phase == 0);
		if (++phase < s->ae.nacc)
			continue;
		phase = 0;

		measure(&st);
		{
			int dg = s->ae.dgain;

			prepare((int)(s->wb_r * dg + 0.5), dg, (int)(s->wb_b * dg + 0.5));
		}
		demosaic_to_yuyv();
		if (write(s->out_fd, out, sizeof(out)) < 0) {
			perror("loopback write");
			goto out;
		}
		s->outputs++;

		/* gray-world white balance on unclipped pixels, smoothed */
		if (st.wb_r > 1000 && st.wb_b > 1000) {
			double tr = st.wb_g / st.wb_r;
			double tb = st.wb_g / st.wb_b;

			if (tr > 4.0) tr = 4.0;
			if (tb > 4.0) tb = 4.0;
			if (tr < 0.25) tr = 0.25;
			if (tb < 0.25) tb = 0.25;
			s->wb_r += 0.1 * (tr - s->wb_r);
			s->wb_b += 0.1 * (tb - s->wb_b);
		}

		if (getenv("SOFTISP_DEBUG") && s->outputs % 30 == 0) {
			double t = now_ms(), dt = (t - s->rate_t0) / 1000;

			fprintf(stderr,
				"raw %.1f fps out %.1f fps truncated %lu meter %.0f exp %d gain %d frames %d dgain %.2f wb %.2f/%.2f\n",
				(s->raw_frames - s->rate_raw0) / dt, (s->outputs - s->rate_out0) / dt,
				s->truncated, st.meter * s->ae.dgain / 1024,
				s->ae.expo, s->ae.gain, s->ae.nacc, s->ae.dgain / 1024.0,
				s->wb_r, s->wb_b);
			s->rate_t0 = t;
			s->rate_raw0 = s->raw_frames;
			s->rate_out0 = s->outputs;
		}

		/* auto exposure every 3 outputs, on centre-weighted luma */
		if (s->outputs % 3 == 0 && !getenv("SOFTISP_NO_AE")) {
			double luma = st.meter * s->ae.dgain / 1024;
			double err = luma > 1.0 ? AE_TARGET / luma : 4.0;

			if (err > 1.0 + AE_TOL || err < 1.0 - AE_TOL) {
				struct ae old = s->ae;
				double have = (double)old.expo * old.gain * old.nacc *
					      old.dgain / 1024;

				ae_allocate(&s->ae, have * pow(err, 0.5));
				if (s->ae.expo != old.expo || s->ae.gain != old.gain)
					ae_apply(s->sd_fd, &s->ae);
				if (s->ae.nacc != old.nacc) {
					build_lin_lut(s->ae.nacc);
					phase = 0;
				}
			}
		}
	}

out:
	{
		int t = V4L2_BUF_TYPE_VIDEO_CAPTURE;

		xioctl(in_fd, VIDIOC_STREAMOFF, &t);
	}
	for (i = 0; i < nbufs; i++)
		if (bufs[i].start != MAP_FAILED)
			munmap(bufs[i].start, bufs[i].length);
	close(in_fd);
	/*
	 * Readers never see a stale picture after a clean stop or across a
	 * sleep. During a recovery the last frame stays up instead, so viewers
	 * see a short pause rather than a black flash.
	 */
	if (ret >= 0)
		write_black(s->out_fd);
	if (ret == 0)
		s->readers = 0;
	return ret;
}

/*
 * The sensor is stopped (or was never running): confirm to the sleep hook,
 * then wait for SIGUSR2, keeping track of readers that come and go meanwhile.
 */
static void sleep_pause(struct isp *s)
{
	double start = now_ms();
	int fd;

	pause_req = 0;
	resume_req = 0;
	fd = open(PAUSE_MARKER, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
	if (fd >= 0)
		close(fd);
	else
		perror(PAUSE_MARKER);
	fprintf(stderr, "ipu4-softisp: sensor stopped for system sleep\n");

	while (!resume_req) {
		struct pollfd pfd = { .fd = s->out_fd, .events = POLLPRI };

		if (now_ms() - start > PAUSE_MAX_MS) {
			fprintf(stderr, "ipu4-softisp: no resume signal; resuming anyway\n");
			break;
		}
		if (poll(&pfd, 1, 1000) > 0 && (pfd.revents & POLLPRI))
			s->readers = dq_readers(s->out_fd, s->readers);
		pause_req = 0;	/* a repeated pre-sleep signal changes nothing */
	}
	resume_req = 0;
	unlink(PAUSE_MARKER);
	fprintf(stderr, "ipu4-softisp: resumed after system sleep (%d reader%s)\n",
		s->readers, s->readers == 1 ? "" : "s");
}

int main(int argc, char **argv)
{
	struct isp s = {
		.in_name = argc > 1 ? argv[1] : "/dev/video-ipu4-raw",
		.wb_r = 1.0,
		.wb_b = 1.0,
	};
	const char *out_name = argc > 2 ? argv[2] : "/dev/video-ipu4";

	s.out_fd = open(out_name, O_RDWR);
	if (s.out_fd < 0) {
		perror(out_name);
		return 1;
	}
	{
		struct v4l2_format f = { .type = V4L2_BUF_TYPE_VIDEO_OUTPUT };

		f.fmt.pix.width = W;
		f.fmt.pix.height = H;
		f.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
		f.fmt.pix.field = V4L2_FIELD_NONE;
		f.fmt.pix.bytesperline = W * 2;
		f.fmt.pix.sizeimage = W * 2 * H;
		if (xioctl(s.out_fd, VIDIOC_S_FMT, &f) < 0) {
			perror("output S_FMT");
			return 1;
		}
	}
	set_output_interval(s.out_fd);
	{
		struct v4l2_event_subscription sub = {
			.type = V4L2_EVENT_PRI_CLIENT_USAGE,
		};

		if (xioctl(s.out_fd, VIDIOC_SUBSCRIBE_EVENT, &sub) < 0) {
			perror("subscribe to v4l2loopback client usage");
			return 1;
		}
	}

	build_gamma_lut();
	build_lin_lut(MIN_ACC);
	s.sd_fd = open_sensor_subdev();
	if (s.sd_fd >= 0) {
		struct v4l2_queryctrl qe = { .id = V4L2_CID_EXPOSURE };
		struct v4l2_queryctrl qg = { .id = V4L2_CID_ANALOGUE_GAIN };
		int vblank = env_int("SOFTISP_VBLANK", DEFAULT_VBLANK);
		int cap = env_int("SOFTISP_EXP_MAX", 0);

		if (set_ctrl(s.sd_fd, V4L2_CID_VBLANK, vblank) < 0)
			fprintf(stderr, "set vblank %d: %s\n", vblank, strerror(errno));
		if (xioctl(s.sd_fd, VIDIOC_QUERYCTRL, &qe) == 0 && qe.maximum > EXP_MIN)
			exp_max = qe.maximum;
		if (cap >= EXP_MIN && cap < exp_max)
			exp_max = cap;
		if (xioctl(s.sd_fd, VIDIOC_QUERYCTRL, &qg) == 0 && qg.maximum > GAIN_MIN)
			gain_max = qg.maximum;
		fprintf(stderr, "ipu4-softisp: vblank %d, exposure limit %d lines, gain limit %d\n",
			vblank, exp_max, gain_max);
	}
	s.ae = (struct ae){ .expo = exp_max, .gain = 1024, .nacc = MIN_ACC, .dgain = 1024 };

	{
		struct sigaction sa = { .sa_handler = on_sleep_signal };

		/* no SA_RESTART: poll() must return EINTR so the loops notice */
		sigemptyset(&sa.sa_mask);
		sigaction(SIGUSR1, &sa, NULL);
		sigaction(SIGUSR2, &sa, NULL);
	}

	/* The first write claims the loopback output, so readers can stream. */
	write_black(s.out_fd);
	fprintf(stderr, "ipu4-softisp: %s -> %s, %dx%d YUYV, on demand\n",
		s.in_name, out_name, W, H);

	for (;;) {
		if (pause_req)
			sleep_pause(&s);
		resume_req = 0;
		if (s.readers <= 0) {
			struct pollfd pfd = { .fd = s.out_fd, .events = POLLPRI };
			int pr = poll(&pfd, 1, IDLE_REFRESH_MS);

			if (pr < 0) {
				if (errno == EINTR)
					continue;
				perror("poll");
				return 1;
			}
			/*
			 * A reader that starts streaming first receives the last frame
			 * written, stamped with the time it was written. Players that
			 * pace frames by timestamp (ffplay, guvcview) hold that frame
			 * for the whole idle period before showing the next one, so
			 * keep it recent.
			 */
			if (pr == 0) {
				write_black(s.out_fd);
				continue;
			}
			if (pfd.revents & POLLPRI)
				s.readers = dq_readers(s.out_fd, s.readers);
			if (s.readers <= 0)
				continue;
		}
		fprintf(stderr, "ipu4-softisp: reader active, starting sensor\n");
		for (int failures = 0;;) {
			unsigned long before = s.raw_frames;
			int r = stream_session(&s);

			if (r >= 0)
				break;
			if (s.raw_frames - before > 600)	/* ran ~10 s: not a streak */
				failures = 0;
			if (++failures >= MAX_RECOVERIES) {
				fprintf(stderr, "ipu4-softisp: capture keeps failing; exiting\n");
				return 1;
			}
			fprintf(stderr, "ipu4-softisp: restarting capture (attempt %d)\n",
				failures + 1);
		}
		if (s.readers <= 0) {
			/* undo the 10000/300000 a Chromium-based reader may have set */
			set_output_interval(s.out_fd);
			fprintf(stderr, "ipu4-softisp: no readers, sensor stopped\n");
		}
	}
}
