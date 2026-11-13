/* sky_env.c — headless driver for the skyroads-sdl game logic, for training pilots.
 *
 * Runs the port's own game() with no window, no audio and no 3D drawing. Once per game frame it
 * calls back into the host (Python, via ctypes) with the ship state; the host returns the keys to
 * hold. A flight ends as soon as the ship crashes, falls or finishes, without the crash animation.
 *
 * Frames: the DOS game reads the controls once per drawn frame and then catches its physics up
 * (36 ticks a second). We measured about 26 frames a second in DOSBox, so by default each frame
 * advances 36/26 ticks on average (1 or 2). The ratio can be changed per flight.
 *
 * Only this file is ours; the port's sources are compiled unchanged. Pattern follows the port's
 * own game_sim.c / assets_headless.c: define the platform stubs, then include assets.c.
 */
#include "assets_platform.h"
#include "platform.h"
#include "render.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern uint32_t x;
extern duint y, z;
extern int32_t vx;
extern int16_t vz;
extern duint crash_type, fuel, oxy;
extern duint road_len;
int game(int the_end);

typedef struct {
    uint32_t x;          /* forward, 65536 per road row */
    uint32_t y;          /* lateral, 128 per pixel; 32768 = road centre */
    uint32_t z;          /* height, 10240 = road level */
    int32_t vx;          /* forward speed, 65536/36 rows per tick per unit... max 10922 */
    int32_t vz;
    uint32_t crash;      /* game's crash_type: 0 none, 1 wall, 2 explosion, 3 hole, 4 fuel, 5 oxygen */
    uint32_t fuel, oxy;
    uint32_t frame, tick;
} env_state;

typedef unsigned (*env_policy)(const env_state *s);

volatile duint Time;
static env_policy policy;
static env_state st;
static unsigned held;            /* keys chosen for the current frame */
static int flying, done, outcome;
static double tick_acc, ticks_per_frame = 36.0 / 26.0;
static uint32_t rng, jitter;
static uint32_t max_frames;

#define OUT_RUNNING 0
#define OUT_FINISHED 1
#define OUT_CRASHED 2           /* crash_type tells which kind */
#define OUT_TIMEOUT 3

const char *cfg_path(void) { return "skyroads_env.cfg"; }
int  plat_pump(void)      { return 1; }
void plat_exit(int c)     { exit(c); }
void plat_present(void)   {}
int  plat_getch(void)     { return 0; }
int  plat_getch_ext(void) { return 0; }
void plat_sleep(int ms)   { (void)ms; }
void plat_osd(const char *msg) { (void)msg; }
void sbdma(const uint8_t *buf, uint32_t len, duint smprate) { (void)buf; (void)len; (void)smprate; }
void sbstop(void) {}

unsigned plat_keys(void) { return done ? K_ESC : (flying ? held : 0); }

void plat_tick_update(void) {
    if (!flying) {
        Time++;                                  /* fades and the start-up frame */
        if (x != 0 && !done) flying = 1;         /* game_body has placed the ship */
        if (!flying) return;
    }
    if (done) { Time++; return; }
    /* the frame that just ended: report it, then choose keys for the next one */
    st.x = x; st.y = y; st.z = z; st.vx = vx; st.vz = vz;
    st.crash = crash_type; st.fuel = fuel; st.oxy = oxy;
    if (crash_type) { outcome = OUT_CRASHED; done = 1; Time++; return; }
    if (z == 0 && x >= (uint32_t)road_len * 65536u - 65536u) {   /* display_road_end zeroes z */
        outcome = OUT_FINISHED; done = 1; Time++; return;
    }
    if (st.frame >= max_frames) { outcome = OUT_TIMEOUT; done = 1; Time++; return; }
    held = policy ? policy(&st) : 0;
    if (held & K_ESC) { outcome = OUT_TIMEOUT; done = 1; }
    st.frame++;
    unsigned n;
    if (jitter) {               /* like DOSBox: each frame covers 1 or 2 ticks, unevenly */
        rng = rng * 1664525u + 1013904223u;
        n = 1u + ((rng >> 8) / 16777216.0 < ticks_per_frame - 1.0);
    } else {
        tick_acc += ticks_per_frame;
        n = (unsigned)tick_acc;
        tick_acc -= n;
    }
    Time = (duint)(Time + n);
    st.tick += n;
}

#include "assets.c"

#ifndef SKYENV_DRAW
/* ---- drawing is not needed for training ---- */
void initvid(void) {}
void video(int vx_, int vy_, int vz_, const uint8_t *car_ptr, int in_tunnel, int rel_z, duint page_seg) {
    (void)vx_; (void)vy_; (void)vz_; (void)car_ptr; (void)in_tunnel; (void)rel_z; (void)page_seg;
}
#endif

/* The current screen (320x200) as RGB bytes; meaningful only in the drawing build (skyenv_draw.dll). */
EXPORT void env_frame_rgb(uint8_t *out) {
    const uint8_t *src = vga_mem();
    for (int i = 0; i < 320 * 200; i++) {
        const uint8_t *c = g_palette[src[i]];
        out[3 * i] = (uint8_t)(c[0] * 255 / 63);
        out[3 * i + 1] = (uint8_t)(c[1] * 255 / 63);
        out[3 * i + 2] = (uint8_t)(c[2] * 255 / 63);
    }
}

/* ---- exported API ---- */
static int loaded_road = -1;

EXPORT int env_init(const char *data_dir) {
    set_data_dir(data_dir);
    load_data();
    load_trekdat();
    if (SysErr) return -1;
    initvid();
    start_alloc();
    load_game_data();
    if (SysErr) return -2;
    return 0;
}

/* nr: 1..30 as in the game's roads.lzs (road 0 is the demo road). Returns road length in rows. */
EXPORT int env_load_road(int nr) {
    if (loaded_road >= 0) free_memory();
    start_alloc();
    road_len = load_road((duint)nr);
    load_background((duint)((nr - 1) / 3));
    if (SysErr) return -1;
    loaded_road = nr;
    return (int)road_len;
}

EXPORT int env_gravity(void) { return (int)gravity; }
EXPORT const duint *env_road_cells(void) { return &Road_Dat[0][0]; }   /* rows of 7 cells */

/* One flight on the loaded road. Returns OUT_*; *out receives the final state.
 * seed 0: frames advance an even 36/26 ticks on average; otherwise 1 or 2 ticks at random. */
EXPORT int env_fly(env_policy p, double tpf, uint32_t frames_cap, uint32_t seed, env_state *out) {
    policy = p;
    rng = seed; jitter = seed != 0;
    ticks_per_frame = tpf > 0 ? tpf : 36.0 / 26.0;
    max_frames = frames_cap ? frames_cap : 36 * 150;
    memset(&st, 0, sizeof st);
    held = 0; flying = 0; done = 0; outcome = OUT_RUNNING; tick_acc = 0;
    x = 0;
    game(0);
    flying = 0;
    if (out) *out = st;
    return outcome;
}
