// hyprhole — every window change goes through a black hole.
//
//   * closing: a black hole forms in the window, the window spirals into it
//     (frame dragging), stretched into streams (tidal forces), its colours
//     feed a glowing accretion disk, then the hole evaporates in a flash and
//     a gravitational-wave ripple;
//   * opening: a hole spits the window out, unwinding, then closes;
//   * moving / resizing / fullscreen / layout changes: the window carries
//     mass: it bends the light around it as it travels (with a twist and
//     colour fringes), and sends a ripple through space;
//   * workspace switch: a black hole opens in the middle of the screen and
//     swallows the old desktop, revealing the new one through its lensing;
//   * dragging (SUPER + mouse): the window collapses into a small black hole
//     that follows the cursor, wrapped in an accretion disk made from the
//     window; dropping it lets the window burst back out in its new place.
//
// Optics: background light is bent by the point-mass lens equation
// (beta = theta - thetaE^2 / theta), giving the Einstein ring and the mirrored
// image around the horizon; red, green and blue use slightly different
// Einstein radii (chromatic fringing); the disk is a thin tilted ring with
// Keplerian shear, Doppler beaming and its far side lensed up over the hole;
// a photon ring hugs the horizon.
//
// Rendering: every frame the damaged part of the composited windows is kept
// in `last` and of the background (before windows) in `bg`. While effects
// run, the real windows under them are painted over with the background and
// each effect is drawn from `last` (live content) or a snapshot (closing
// windows, the dragged window, the old desktop). Hyprland's own window and
// workspace animations must be disabled.

#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/desktop/Workspace.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>
#include <hyprland/src/helpers/math/Math.hpp>
#include <hyprland/src/layout/LayoutManager.hpp>
#include <hyprland/src/pointer/PointerManager.hpp>
#include <hyprland/src/managers/input/InputManager.hpp>

#include <GLES3/gl32.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

static HANDLE PHANDLE = nullptr; // not inline: a GNU-unique symbol would pin the .so on unload

namespace {
    constexpr double EDGE       = 6.0;  // window boxes grown by this to take borders along, logical px
    constexpr double WIN_CORNER = 10.0; // window corner radius, logical px
    // timing (s)
    constexpr float T_CLOSE = 0.80f;
    constexpr float T_OPEN  = 0.70f;
    constexpr float T_MOVE  = 0.45f;
    constexpr float T_WS    = 0.90f;
    constexpr float T_PICK  = 0.40f; // drag: window -> hole
    constexpr float T_DROP  = 0.60f; // drag: hole -> window
    // holes
    constexpr double HOLE_MIN  = 16.0, HOLE_MAX = 64.0; // horizon radius of a window's hole, logical px
    constexpr double DRAG_HOLE = 30.0;                  // the dragged window's hole, logical px
    constexpr float  EINSTEIN  = 2.4f;                  // Einstein radius / horizon radius

    using Clock = std::chrono::steady_clock;

    struct V2 {
        float x = 0, y = 0;
    };
    inline V2 operator+(V2 a, V2 b) {
        return {a.x + b.x, a.y + b.y};
    }
    inline V2 operator-(V2 a, V2 b) {
        return {a.x - b.x, a.y - b.y};
    }
    inline V2 operator*(V2 a, float s) {
        return {a.x * s, a.y * s};
    }
    inline float len(V2 a) {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }
    float smooth01(float t) {
        t = std::clamp(t, 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }
    float easeInOut(float t) {
        t = std::clamp(t, 0.f, 1.f);
        return t < 0.5f ? 4.f * t * t * t : 1.f - std::pow(-2.f * t + 2.f, 3.f) / 2.f;
    }
    float secsSince(Clock::time_point t) {
        return std::chrono::duration<float>(Clock::now() - t).count();
    }
    CBox lerpBox(const CBox& a, const CBox& b, float k) {
        return {a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.w + (b.w - a.w) * k, a.h + (b.h - a.h) * k};
    }
    V2 centreOf(const CBox& b) {
        return {(float)(b.x + b.w * 0.5), (float)(b.y + b.h * 0.5)};
    }

    enum eKind : uint8_t {
        CLOSE,
        OPEN,
        MOVE,
        WORKSPACE,
        DRAG
    };

    // What the shader gets for one effect, this frame.
    struct SParams {
        V2    c;               // hole / mass centre, GL px
        CBox  win;             // the window in flat space, GL px
        float s      = 1.f;    // how much of the window is left (1 whole .. 0 swallowed)
        float rh     = 0.f;    // horizon radius, px
        float E      = 0.f;    // Einstein radius, px
        float twist  = 0.f;    // frame dragging
        float disk   = 0.f;    // accretion disk brightness
        float flashR = 0.f, flashA = 0.f; // gravitational-wave ring
        float lensWin = 1.f;   // the window itself is lensed (it is falling in), or only the space around it
        float corner  = 0.f;
        float alpha   = 1.f;   // the window's opacity
        V2    vel;             // px per second (colour fringes along the motion)
    };

    struct SEffect {
        eKind                    kind = MOVE;
        uintptr_t                id   = 0;
        PHLWINDOWREF             win;
        bool                     hidden = false, released = false, dead = false;
        bool                     needSnap = false; // copy the picture from `last` this frame
        CBox                     snapBox;          // GL px
        SP<Render::IFramebuffer> snap;             // own picture; none: live from `last` at `live`
        CBox                     live;             // GL px: where the real window is (content, painted over)
        CBox                     from, to;         // GL px
        Clock::time_point        start, lastStep;
        float                    dur = 0.5f, spin = 1.f, rhPeak = 30.f;
        V2                       hole, holeV, grabOff; // drag: hole position, velocity; window rect offset from it
        V2                       prevC;
        SParams                  p;
    };

    struct SWinBox {
        uintptr_t    id;
        PHLWINDOWREF win;
        CBox         box; // logical, grown by EDGE
    };

    struct SMonitorState {
        SP<Render::IFramebuffer> last, bg, scene;
        std::vector<std::unique_ptr<SEffect>> fx;
        std::vector<SWinBox>                  cached;
        WORKSPACEID                           cachedWs = WORKSPACE_INVALID;
        float                                 scale    = 1.f;
        bool                                  lastValid = false;
    };

    std::unordered_map<MONITORID, SMonitorState> g_mon;
    std::vector<CHyprSignalListener>            g_listeners;

    // ---------------------------------------------------------------- GL ---

    const char* FULL_VERT = R"(#version 300 es
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

    const char* HOLE_FRAG = R"(#version 300 es
precision highp float;
uniform sampler2D scene;     // space behind (what the light comes from)
uniform sampler2D content;   // the window's picture
uniform vec4  contentUv;     // where the window is in `content`
uniform vec4  winBox;        // the window in flat space, GL px
uniform float corner;
uniform vec2  c;             // hole centre, GL px
uniform float s;             // how much of the window is left
uniform float rh;            // horizon radius, px
uniform float E;             // Einstein radius, px
uniform float twist;
uniform float disk;
uniform float flashR;
uniform float flashA;
uniform float lensWin;
uniform float winAlpha;
uniform float spin;          // +1 / -1: which way it turns
uniform float reach;         // beyond this the effect fades out (px)
uniform vec2  vel;
uniform float time;
uniform float px;            // 1 logical px
uniform vec2  fbSize;
out vec4 frag;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
    return v;
}
vec2 rot(vec2 v, float a) { float cs = cos(a), sn = sin(a); return vec2(cs * v.x - sn * v.y, sn * v.x + cs * v.y); }
float sdRound(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }

// hot -> cool: white, gold, orange, deep red
vec3 blackbody(float k) {
    k = clamp(k, 0.0, 1.0);
    vec3 a = mix(vec3(1.0, 0.97, 0.92), vec3(1.0, 0.78, 0.40), smoothstep(0.0, 0.3, k));
    a = mix(a, vec3(1.0, 0.45, 0.10), smoothstep(0.3, 0.65, k));
    return mix(a, vec3(0.55, 0.08, 0.03), smoothstep(0.65, 1.0, k));
}

float L0() { return max(max(winBox.z, winBox.w) * 0.35, 1.0); }

// Where a point that is now at screen `q` was on the window before it fell:
// undo the frame-dragging swirl and the infall (inner parts fall faster,
// which tears the window into spiral streams).
vec4 windowAt(vec2 q) {
    vec2  rel = q - c;
    float r1  = length(rel);
    float L   = L0();
    float th  = atan(rel.y, rel.x) - spin * twist / (r1 / L + 0.25);
    float ss  = max(s, 1e-3);
    float r0  = r1 / ss;
    for (int i = 0; i < 3; i++)
        r0 = r1 / pow(ss, 1.0 + 2.5 * L * 0.2 / (r0 + L * 0.2));
    vec2 src = c + vec2(cos(th), sin(th)) * r0;
    vec2 uv  = (src - winBox.xy) / winBox.zw;
    float m  = 1.0 - smoothstep(-1.0, 1.0, sdRound(src - winBox.xy - winBox.zw * 0.5, winBox.zw * 0.5, corner));
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0)
        m = 0.0;
    vec3 col = texture(content, contentUv.xy + clamp(uv, 0.0, 1.0) * contentUv.zw).rgb;
    // light climbing out near the hole is redshifted and dimmed
    if (rh > 0.0) {
        float g = smoothstep(rh, rh * 3.0, r1);
        col = mix(col * vec3(1.0, 0.35, 0.18) * 0.6, col, g);
        m *= smoothstep(rh * 0.98, rh * 1.25, r1);
    }
    return vec4(col, m * winAlpha);
}

// the bent position of the light arriving at screen point p, for Einstein radius e
vec2 lens(vec2 p, float e) {
    vec2  d = p - c;
    float r = max(length(d), 1e-3);
    float fall = 1.0 - smoothstep(reach * 0.55, reach, r);
    float b = r - e * e / r * fall;
    // frame dragging drags the background round too, close to the hole
    vec2 dir = rot(d / r, spin * twist * 0.35 * rh * rh / (r * r + rh * rh) * 4.0);
    return c + dir * b;
}

void main() {
    vec2  p = gl_FragCoord.xy;
    vec2  d = p - c;
    float r = length(d);
    float fade = 1.0 - smoothstep(reach * 0.85, reach, r);

    // a gravitational wave: space ripples outward from the hole
    float k = (r - flashR) / (16.0 * px);
    if (flashA > 0.0 && r > 1e-3)
        p += d / r * flashA * 9.0 * px * sin(k * 2.2) * exp(-k * k * 0.18);

    // background, lensed, each colour bent a little differently
    vec3 col;
    vec4 w;
    if (lensWin > 0.5) {
        vec4 wr = windowAt(lens(p, E * 1.035));
        vec4 wg = windowAt(lens(p, E));
        vec4 wb = windowAt(lens(p, E * 0.965));
        col.r = mix(texture(scene, lens(p, E * 1.035) / fbSize).r, wr.r, wr.a);
        col.g = mix(texture(scene, lens(p, E) / fbSize).g, wg.g, wg.a);
        col.b = mix(texture(scene, lens(p, E * 0.965) / fbSize).b, wb.b, wb.a);
    } else {
        // a moving mass: space around it bends, the window keeps its shape but
        // twists a little and fringes along its motion
        col = vec3(texture(scene, lens(p, E * 1.06) / fbSize).r, texture(scene, lens(p, E) / fbSize).g, texture(scene, lens(p, E * 0.94) / fbSize).b);
        vec2 f = vel * 0.012;
        vec4 wr = windowAt(p - f), wg = windowAt(p), wb = windowAt(p + f);
        col = vec3(mix(col.r, wr.r, wr.a), mix(col.g, wg.g, wg.a), mix(col.b, wb.b, wb.a));
    }

    if (rh > 0.0) {
        float rr = r / rh;
        // the event horizon
        float hz = 1.0 - smoothstep(0.97, 1.0, rr);
        col = mix(col, vec3(0.0), hz);

        // accretion disk: a thin tilted ring (front side crosses the hole),
        // its far side lensed up over the top as a halo
        vec2  dp  = vec2(d.x, d.y / 0.23) / rh;
        float rd  = length(dp);
        float ad  = atan(dp.y, dp.x) + spin * time * 2.4 / pow(max(rd, 1.0), 1.5);
        float bands = 0.35 + 0.9 * fbm(vec2(cos(ad), sin(ad)) * rd * 1.6 + vec2(rd * 2.0, 0.0));
        float dopp  = 1.0 + 0.75 * spin * (-dp.x / max(rd, 1e-3));
        float kk    = clamp((rd - 1.5) / 3.3, 0.0, 1.0);
        vec3  feed  = texture(content, contentUv.xy + vec2(fract(ad / 6.2831853), kk) * contentUv.zw).rgb;
        vec3  dcol  = mix(blackbody(kk), feed * 1.7 + 0.15, 0.40);
        float front = smoothstep(1.45, 1.9, rd) * (1.0 - smoothstep(3.4, 4.8, rd));
        float hidden = (d.y < 0.0 && rr < 1.0) ? 0.0 : 1.0; // behind the hole on the far side
        col += dcol * front * bands * dopp * 1.5 * disk * hidden;

        float ah   = atan(d.y, d.x) + spin * time * 1.6;
        float hb   = 0.35 + 0.9 * fbm(vec2(cos(ah), sin(ah)) * 2.4 + rr * 3.0);
        float halo = smoothstep(1.02, 1.18, rr) * (1.0 - smoothstep(1.35, 2.2, rr));
        float top  = 0.55 + 0.9 * max(-d.y / max(r, 1e-3), 0.0); // rows run top-down: the arc over the top
        col += blackbody(0.25 + 0.4 * (rr - 1.0)) * halo * hb * top * 1.3 * disk;
        // photon ring
        col += vec3(1.0, 0.92, 0.82) * exp(-pow((rr - 1.045) / 0.022, 2.0)) * 1.6 * disk;
    }

    // the flash of the ripple, split into colours
    if (flashA > 0.0) {
        float w = 6.0 * px;
        col += vec3(exp(-pow((r - flashR - 2.0 * px) / w, 2.0)), exp(-pow((r - flashR) / w, 2.0)), exp(-pow((r - flashR + 2.0 * px) / w, 2.0)))
             * vec3(1.0, 0.9, 0.95) * flashA * 0.8;
    }

    vec3 base = texture(scene, gl_FragCoord.xy / fbSize).rgb;
    frag = vec4(mix(base, col, fade), 1.0);
}
)";

    struct SGL {
        GLuint prog = 0, vao = 0;
        GLint  uScene = -1, uContent = -1, uContentUv = -1, uWin = -1, uCorner = -1, uC = -1, uS = -1, uRh = -1, uE = -1, uTwist = -1, uDisk = -1, uFlashR = -1,
              uFlashA = -1, uLensWin = -1, uWinAlpha = -1, uSpin = -1, uReach = -1, uVel = -1, uTime = -1, uPx = -1, uFb = -1;
        bool failed = false;
    } g_gl;

    GLuint compile(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024] = {};
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            HyprlandAPI::addNotification(PHANDLE, std::string("[hyprhole] shader: ") + log, CHyprColor{1, 0.2, 0.2, 1}, 8000);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    GLuint link(const char* vsrc, const char* fsrc) {
        GLuint vs = compile(GL_VERTEX_SHADER, vsrc), fs = compile(GL_FRAGMENT_SHADER, fsrc);
        if (!vs || !fs) {
            if (vs)
                glDeleteShader(vs);
            if (fs)
                glDeleteShader(fs);
            return 0;
        }
        GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    bool ensureGL() {
        if (g_gl.prog)
            return true;
        if (g_gl.failed)
            return false;
        g_gl.prog = link(FULL_VERT, HOLE_FRAG);
        if (!g_gl.prog) {
            g_gl.failed = true;
            return false;
        }
        const auto U    = [](const char* n) { return glGetUniformLocation(g_gl.prog, n); };
        g_gl.uScene     = U("scene");
        g_gl.uContent   = U("content");
        g_gl.uContentUv = U("contentUv");
        g_gl.uWin       = U("winBox");
        g_gl.uCorner    = U("corner");
        g_gl.uC         = U("c");
        g_gl.uS         = U("s");
        g_gl.uRh        = U("rh");
        g_gl.uE         = U("E");
        g_gl.uTwist     = U("twist");
        g_gl.uDisk      = U("disk");
        g_gl.uFlashR    = U("flashR");
        g_gl.uFlashA    = U("flashA");
        g_gl.uLensWin   = U("lensWin");
        g_gl.uWinAlpha  = U("winAlpha");
        g_gl.uSpin      = U("spin");
        g_gl.uReach     = U("reach");
        g_gl.uVel       = U("vel");
        g_gl.uTime      = U("time");
        g_gl.uPx        = U("px");
        g_gl.uFb        = U("fbSize");
        GLint prevVao = 0;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
        glGenVertexArrays(1, &g_gl.vao); // empty: the vertex shader makes the triangle
        glBindVertexArray(prevVao);
        return true;
    }

    std::optional<GLuint> fbID(const SP<Render::IFramebuffer>& fb) {
        auto* gl = dynamic_cast<Render::GL::CGLFramebuffer*>(fb.get());
        if (!gl)
            return std::nullopt;
        return gl->getFBID();
    }

    bool ensureFB(SP<Render::IFramebuffer>& fb, int w, int h, DRMFormat fmt, const char* name) {
        if (fb && fb->m_size == Vector2D{(double)w, (double)h} && fb->m_drmFormat == fmt)
            return true;
        if (!fb)
            fb = g_pHyprRenderer->createFB(name);
        return fb->alloc(w, h, fmt);
    }

    bool ensureFB(SP<Render::IFramebuffer>& fb, const SP<Render::IFramebuffer>& like, const char* name) {
        return ensureFB(fb, (int)like->m_size.x, (int)like->m_size.y, like->m_drmFormat, name);
    }

    // Copy rect (x0,y0)-(x1,y1) of `from` to (dx,dy) in `to`. Leaves bindings and scissor as they were.
    void blitRect(const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to, int x0, int y0, int x1, int y1, int dx, int dy) {
        const auto A = fbID(from), B = fbID(to);
        if (!A || !B || x1 <= x0 || y1 <= y0)
            return;
        GLint prevRead = 0, prevDraw = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
        const GLboolean SCISSOR = glIsEnabled(GL_SCISSOR_TEST);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, *A);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, *B);
        glBlitFramebuffer(x0, y0, x1, y1, dx, dy, dx + (x1 - x0), dy + (y1 - y0), GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);
        if (SCISSOR)
            glEnable(GL_SCISSOR_TEST);
    }

    void blitAll(const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to) {
        blitRect(from, to, 0, 0, (int)from->m_size.x, (int)from->m_size.y, 0, 0);
    }

    // The render buffer only holds valid pixels inside this frame's damage:
    // copy just those rects into a persistent buffer.
    void blitDamage(const PHLMONITOR& mon, const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to) {
        CRegion region = g_pHyprRenderer->m_renderData.damage.copy();
        region.intersect(CBox{{}, mon->m_transformedSize});
        region.transform(Math::wlTransformToHyprutils(Math::invertTransform(mon->m_transform)), mon->m_transformedSize.x, mon->m_transformedSize.y);
        region.intersect(CBox{{}, from->m_size});
        for (const auto& r : region.getRects())
            blitRect(from, to, r.x1, r.y1, r.x2, r.y2, r.x1, r.y1);
    }

    CBox glBox(const CBox& b, const PHLMONITOR& mon) {
        const double S = mon->m_scale;
        return CBox{(b.x - mon->m_position.x) * S, (b.y - mon->m_position.y) * S, b.w * S, b.h * S}; // rows run top-down
    }

    V2 cursorGL(const PHLMONITOR& mon) {
        const Vector2D C = Pointer::mgr()->position();
        return {(float)((C.x - mon->m_position.x) * mon->m_scale), (float)((C.y - mon->m_position.y) * mon->m_scale)};
    }

    // ------------------------------------------------------------ effects ---

    float holeFor(const SMonitorState& st, const CBox& b) {
        return std::clamp((float)std::min(b.w, b.h) * 0.09f, (float)(HOLE_MIN * st.scale), (float)(HOLE_MAX * st.scale));
    }

    SEffect& addFx(SMonitorState& st, eKind kind, uintptr_t id, float dur) {
        auto F   = std::make_unique<SEffect>();
        F->kind  = kind;
        F->id    = id;
        F->dur   = dur;
        F->start = F->lastStep = Clock::now();
        F->spin  = (id >> 4) & 1 ? 1.f : -1.f;
        st.fx.push_back(std::move(F));
        return *st.fx.back();
    }

    SEffect* findFx(SMonitorState& st, uintptr_t id) {
        for (auto& f : st.fx)
            if (f->id == id && !f->dead && (f->kind == OPEN || f->kind == MOVE || f->kind == DRAG))
                return f.get();
        return nullptr;
    }

    // Advance one effect; false once it has finished.
    bool step(SMonitorState& st, SEffect& f, const PHLMONITOR& mon) {
        const float T  = secsSince(f.start);
        const float t  = std::clamp(T / f.dur, 0.f, 1.f);
        const float DT = std::min(secsSince(f.lastStep), 1.f / 30.f);
        f.lastStep     = Clock::now();
        SParams& p     = f.p;
        p              = {};
        p.corner       = (float)(WIN_CORNER * st.scale);
        const float S  = st.scale;

        switch (f.kind) {
            case CLOSE: {
                p.win    = f.from;
                p.c      = centreOf(f.from);
                p.s      = 1.f - smooth01(t / 0.8f);
                p.rh     = f.rhPeak * smooth01(t / 0.3f) * (1.f - smooth01((t - 0.74f) / 0.2f));
                p.twist  = 2.6f * smooth01(t / 0.8f);
                p.disk   = smooth01(t / 0.25f) * (1.f - smooth01((t - 0.66f) / 0.22f));
                if (t > 0.76f) {
                    const float K = (t - 0.76f) / 0.24f;
                    p.flashR      = K * (float)std::max(f.from.w, f.from.h) * 0.9f;
                    p.flashA      = 1.f - K;
                }
                break;
            }
            case OPEN: {
                const float u = 1.f - t; // a close, played backwards
                p.win    = f.to;
                p.c      = centreOf(f.to);
                p.s      = 1.f - smooth01(u / 0.8f);
                p.rh     = f.rhPeak * smooth01(u / 0.3f) * (1.f - smooth01((u - 0.8f) / 0.2f));
                p.twist  = 2.6f * smooth01(u / 0.8f);
                p.disk   = smooth01(u / 0.25f) * (1.f - smooth01((u - 0.8f) / 0.2f));
                if (t < 0.3f) {
                    const float K = t / 0.3f;
                    p.flashR      = K * (float)std::max(f.to.w, f.to.h) * 0.7f;
                    p.flashA      = (1.f - K) * 0.8f;
                }
                break;
            }
            case MOVE: {
                const float K = easeInOut(t);
                p.win     = lerpBox(f.from, f.to, K);
                p.c       = centreOf(p.win);
                p.lensWin = 0.f;
                const float B = std::sin(3.14159265f * t);
                p.E       = 0.42f * (float)std::min(p.win.w, p.win.h) * 0.5f * B;
                p.twist   = 0.06f * B;
                p.flashR  = t * (float)std::max(p.win.w, p.win.h) * 1.1f;
                p.flashA  = (1.f - t) * 0.55f;
                const V2 C = p.c;
                p.vel      = DT > 0.f ? (C - f.prevC) * (1.f / DT) : V2{};
                f.prevC    = C;
                break;
            }
            case WORKSPACE: {
                p.win     = f.from; // the old desktop, whole
                p.corner  = 0.f;
                p.c       = centreOf(f.from);
                p.s       = 1.f - smooth01(t / 0.78f);
                p.rh      = f.rhPeak * smooth01(t / 0.3f) * (1.f - smooth01((t - 0.72f) / 0.24f));
                p.twist   = 3.2f * smooth01(t / 0.8f);
                p.disk    = smooth01(t / 0.22f) * (1.f - smooth01((t - 0.62f) / 0.26f));
                if (t > 0.74f) {
                    const float K = (t - 0.74f) / 0.26f;
                    p.flashR      = K * (float)std::max(f.from.w, f.from.h) * 0.75f;
                    p.flashA      = 1.f - K;
                }
                break;
            }
            case DRAG: {
                const V2 CUR = cursorGL(mon);
                if (!f.released) {
                    // the hole follows the cursor on a spring (it has mass)
                    const V2 A = (CUR - f.hole) * 600.f - f.holeV * 38.f;
                    f.holeV    = f.holeV + A * DT;
                    f.hole     = f.hole + f.holeV * DT;
                    const float K = std::clamp(T / T_PICK, 0.f, 1.f);
                    p.c      = f.hole;
                    p.win    = CBox{f.from.x + (CUR.x - f.grabOff.x), f.from.y + (CUR.y - f.grabOff.y), f.from.w, f.from.h};
                    p.s      = 1.f - smooth01(K);
                    p.rh     = (float)(DRAG_HOLE * S) * smooth01(K / 0.8f);
                    p.twist  = 2.8f * smooth01(K);
                    p.disk   = smooth01(K / 0.7f);
                    p.vel    = f.holeV;
                } else {
                    const float K = std::clamp(T / T_DROP, 0.f, 1.f);
                    const float M = easeInOut(std::min(1.f, K / 0.6f));
                    p.c      = f.hole + (centreOf(f.to) - f.hole) * M;
                    p.win    = CBox{p.c.x - f.to.w * 0.5, p.c.y - f.to.h * 0.5, f.to.w, f.to.h};
                    p.s      = smooth01(K / 0.85f);
                    p.rh     = (float)(DRAG_HOLE * S) * (1.f - smooth01(K / 0.7f));
                    p.twist  = 2.8f * (1.f - smooth01(K / 0.85f));
                    p.disk   = 1.f - smooth01(K / 0.6f);
                    if (K < 0.35f) {
                        p.flashR = K / 0.35f * (float)std::max(f.to.w, f.to.h) * 0.6f;
                        p.flashA = (1.f - K / 0.35f) * 0.7f;
                    }
                    if (T >= T_DROP)
                        return false;
                }
                break;
            }
        }
        p.E = p.rh > 0.f ? p.rh * EINSTEIN : p.E;
        return f.kind == DRAG || T < f.dur;
    }

    // ---------------------------------------------------- window tracking ---

    std::vector<SWinBox> windowBoxes(const PHLWORKSPACE& ws, const PHLWINDOW& skip) {
        std::vector<SWinBox> out;
        for (const auto& w : Desktop::windowState()->windows()) {
            if (!w || !w->m_isMapped || w->isHidden() || w->m_workspace != ws || w == skip)
                continue;
            out.push_back({(uintptr_t)w.get(), w, CBox{w->positionAnimation()->goal(), w->sizeAnimation()->goal()}.expand(EDGE)});
        }
        return out;
    }

    const SWinBox* findBox(const std::vector<SWinBox>& v, uintptr_t id) {
        for (const auto& b : v)
            if (b.id == id)
                return &b;
        return nullptr;
    }

    bool sameBox(const CBox& a, const CBox& b) {
        return std::abs(a.x - b.x) < 0.5 && std::abs(a.y - b.y) < 0.5 && std::abs(a.w - b.w) < 0.5 && std::abs(a.h - b.h) < 0.5;
    }

    PHLWINDOW movedWindow(const PHLMONITOR& mon) {
        if (!g_layoutManager)
            return nullptr;
        const auto& DC = g_layoutManager->dragController();
        if (!DC || DC->mode() != MBIND_MOVE)
            return nullptr;
        const auto T = DC->target();
        const auto W = T ? T->window() : nullptr;
        if (!W || !W->m_workspace || W->m_workspace != mon->m_activeWorkspace)
            return nullptr;
        return W;
    }

    void addOpen(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w) {
        auto& f  = addFx(st, OPEN, w.id, T_OPEN);
        f.win    = w.win;
        f.to     = glBox(w.box, mon);
        f.live   = f.to;
        f.rhPeak = holeFor(st, f.to);
    }

    void addClose(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w) {
        auto& f    = addFx(st, CLOSE, w.id, T_CLOSE);
        f.from     = glBox(w.box, mon);
        f.rhPeak   = holeFor(st, f.from);
        f.needSnap = true;
        f.snapBox  = f.from;
    }

    void addMove(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w, const CBox& fromL) {
        const CBox TO = glBox(w.box, mon);
        if (auto* f = findFx(st, w.id)) {
            if (f->kind == DRAG)
                return;
            // already travelling: carry on from where it is drawn now
            f->from  = f->kind == MOVE ? f->p.win : f->to;
            f->kind  = MOVE;
            f->to    = TO;
            f->live  = TO;
            f->dur   = T_MOVE;
            f->start = Clock::now();
            f->prevC = centreOf(f->from);
            return;
        }
        auto& f = addFx(st, MOVE, w.id, T_MOVE);
        f.win   = w.win;
        f.from  = glBox(fromL, mon);
        f.to    = TO;
        f.live  = TO;
        f.prevC = centreOf(f.from);
    }

    // Before the windows are drawn: compare the windows with last frame's.
    void watch(const PHLMONITOR& mon) {
        auto&      st = g_mon[mon->m_id];
        st.scale      = (float)mon->m_scale;
        const auto WS = mon->m_activeWorkspace;
        if (!WS || WS->m_isSpecialWorkspace)
            return;
        const auto DW    = movedWindow(mon);
        const auto BOXES = windowBoxes(WS, DW ? DW : PHLWINDOW{});

        if (st.cachedWs == WORKSPACE_INVALID || !st.lastValid) {
            st.cachedWs = WS->m_id;
            st.cached   = BOXES;
            return;
        }
        if (WS->m_id != st.cachedWs) {
            // workspace switch: a black hole swallows the old desktop
            for (auto& f : st.fx)
                if (f->kind != DRAG)
                    f->dead = true;
            auto& f    = addFx(st, WORKSPACE, 0, T_WS);
            f.from     = CBox{0, 0, (double)mon->m_pixelSize.x, (double)mon->m_pixelSize.y};
            f.rhPeak   = (float)mon->m_pixelSize.y * 0.085f;
            f.spin     = WS->m_id > st.cachedWs ? 1.f : -1.f;
            f.needSnap = true;
            f.snapBox  = f.from;
            st.cachedWs = WS->m_id;
            st.cached   = BOXES;
            g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        bool changed = false;
        for (const auto& w : BOXES) {
            const auto OLD = findBox(st.cached, w.id);
            if (!OLD) {
                if (auto* f = findFx(st, w.id); f && f->kind == DRAG)
                    continue; // a dropped window: its drag effect lets it out
                addOpen(st, mon, w);
                changed = true;
            } else if (!sameBox(OLD->box, w.box)) {
                addMove(st, mon, w, OLD->box);
                changed = true;
            }
        }
        for (const auto& w : st.cached)
            if (!findBox(BOXES, w.id)) {
                if (DW && w.id == (uintptr_t)DW.get())
                    continue; // picked up: the drag takes it
                if (auto* f = findFx(st, w.id))
                    f->dead = true;
                addClose(st, mon, w);
                changed = true;
            }
        st.cached = BOXES;
        if (changed)
            g_pHyprRenderer->damageMonitor(mon);
    }

    // ------------------------------------------------------------ drawing ---

    struct SGLState {
        GLint     prog = 0, vao = 0, fb = 0, active = 0, tex[2] = {}, vp[4] = {}, box[4] = {};
        GLboolean blend = 0, scissor = 0;
        void      save() {
            glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
            for (int u = 0; u < 2; ++u) {
                glActiveTexture(GL_TEXTURE0 + u);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex[u]);
            }
            glGetIntegerv(GL_VIEWPORT, vp);
            glGetIntegerv(GL_SCISSOR_BOX, box);
            blend   = glIsEnabled(GL_BLEND);
            scissor = glIsEnabled(GL_SCISSOR_TEST);
        }
        void restore() {
            for (int u = 1; u >= 0; --u) {
                glActiveTexture(GL_TEXTURE0 + u);
                glBindTexture(GL_TEXTURE_2D, tex[u]);
            }
            glActiveTexture(active);
            glUseProgram(prog);
            glBindVertexArray(vao);
            glBindFramebuffer(GL_FRAMEBUFFER, fb);
            glViewport(vp[0], vp[1], vp[2], vp[3]);
            glScissor(box[0], box[1], box[2], box[3]);
            blend ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
            scissor ? glEnable(GL_SCISSOR_TEST) : glDisable(GL_SCISSOR_TEST);
        }
    };

    void linearClamp(const SP<Render::IFramebuffer>& fb) {
        fb->getTexture()->bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void drawFx(SMonitorState& st, SEffect& f, const SP<Render::IFramebuffer>& target, bool refreshScene) {
        const auto& CONTENT = f.snap ? f.snap : st.last;
        if (!CONTENT)
            return;
        const float FW = (float)target->m_size.x, FH = (float)target->m_size.y;
        const auto& p  = f.p;
        // how far the effect reaches: the window, the disk, the lensing, the ripple
        float reach = 0.f;
        for (const V2 K : {V2{0, 0}, V2{1, 0}, V2{0, 1}, V2{1, 1}})
            reach = std::max(reach, len(V2{(float)(p.win.x + p.win.w * K.x), (float)(p.win.y + p.win.h * K.y)} - p.c));
        reach = std::max({reach * (p.lensWin > 0.5f ? 1.f : 1.15f) + 40.f * st.scale, p.rh * 6.f, p.E * 4.f, p.flashR + 60.f * st.scale});
        if (f.kind == WORKSPACE)
            reach = std::max(FW, FH) * 2.f;

        if (refreshScene && st.scene) // effects stack: each sees the ones drawn before it
            blitAll(target, st.scene);
        glActiveTexture(GL_TEXTURE0);
        linearClamp(st.scene);
        glActiveTexture(GL_TEXTURE1);
        linearClamp(CONTENT);

        glUseProgram(g_gl.prog);
        glUniform1i(g_gl.uScene, 0);
        glUniform1i(g_gl.uContent, 1);
        if (f.snap)
            glUniform4f(g_gl.uContentUv, 0.f, 0.f, 1.f, 1.f);
        else
            glUniform4f(g_gl.uContentUv, (float)(f.live.x / FW), (float)(f.live.y / FH), (float)(f.live.w / FW), (float)(f.live.h / FH));
        glUniform4f(g_gl.uWin, (float)p.win.x, (float)p.win.y, (float)p.win.w, (float)p.win.h);
        glUniform1f(g_gl.uCorner, p.corner);
        glUniform2f(g_gl.uC, p.c.x, p.c.y);
        glUniform1f(g_gl.uS, p.s);
        glUniform1f(g_gl.uRh, p.rh);
        glUniform1f(g_gl.uE, p.E);
        glUniform1f(g_gl.uTwist, p.twist);
        glUniform1f(g_gl.uDisk, p.disk);
        glUniform1f(g_gl.uFlashR, p.flashR);
        glUniform1f(g_gl.uFlashA, p.flashA);
        glUniform1f(g_gl.uLensWin, p.lensWin);
        glUniform1f(g_gl.uWinAlpha, p.alpha);
        glUniform1f(g_gl.uSpin, f.spin);
        glUniform1f(g_gl.uReach, reach);
        glUniform2f(g_gl.uVel, p.vel.x, p.vel.y);
        glUniform1f(g_gl.uTime, secsSince(f.start));
        glUniform1f(g_gl.uPx, st.scale);
        glUniform2f(g_gl.uFb, FW, FH);

        const int X0 = std::clamp((int)std::floor(p.c.x - reach), 0, (int)FW), X1 = std::clamp((int)std::ceil(p.c.x + reach), 0, (int)FW);
        const int Y0 = std::clamp((int)std::floor(p.c.y - reach), 0, (int)FH), Y1 = std::clamp((int)std::ceil(p.c.y + reach), 0, (int)FH);
        if (X1 <= X0 || Y1 <= Y0)
            return;
        glEnable(GL_SCISSOR_TEST);
        glScissor(X0, Y0, X1 - X0, Y1 - Y0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_SCISSOR_TEST);
    }

    // ------------------------------------------------------- pass elements ---

    bool ownMonitorFrame() {
        const auto& rd = g_pHyprRenderer->m_renderData;
        if (g_pHyprRenderer->m_bRenderingSnapshot || rd.projectionType != Render::RPT_MONITOR)
            return false;
        if (rd.mainFB && rd.currentFB != rd.mainFB)
            return false;
        return !(rd.renderModif.enabled && !rd.renderModif.modifs.empty());
    }

    void showWindow(SEffect& f) {
        if (const auto W = f.win.lock())
            W->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(1.f);
        f.hidden = false;
    }

    void startDrag(SMonitorState& st, const PHLMONITOR& mon, const PHLWINDOW& w, const SP<Render::IFramebuffer>& cur) {
        const CBox B  = glBox(CBox{w->positionAnimation()->value(), w->sizeAnimation()->value()}.expand(EDGE), mon);
        const int  X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
        const int  X1 = std::min((int)cur->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)cur->m_size.y, (int)(B.y + B.h));
        SP<Render::IFramebuffer> snap;
        if (X1 - X0 < 8 || Y1 - Y0 < 8 || !ensureFB(snap, X1 - X0, Y1 - Y0, cur->m_drmFormat, "hyprhole-drag"))
            return;
        blitRect(st.last, snap, X0, Y0, X1, Y1, 0, 0);
        for (auto& f : st.fx)
            if (f->id == (uintptr_t)w.get())
                f->dead = true;
        auto& f   = addFx(st, DRAG, (uintptr_t)w.get(), 1.f);
        f.win     = w;
        f.snap    = snap;
        f.from    = CBox{(double)X0, (double)Y0, (double)(X1 - X0), (double)(Y1 - Y0)};
        f.hole    = cursorGL(mon);
        f.grabOff = f.hole; // the window rect moves with the cursor from here
        f.spin    = 1.f;
        w->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(0.f); // the hole stands in for it
        f.hidden = true;
    }

    void releaseDrag(SEffect& f, const PHLMONITOR& mon) {
        const auto W = f.win.lock();
        if (!W) {
            f.dead = true;
            return;
        }
        f.to       = glBox(CBox{W->positionAnimation()->goal(), W->sizeAnimation()->goal()}.expand(EDGE), mon);
        f.released = true;
        f.start    = Clock::now();
    }

    class CElement : public IPassElement {
      public:
        enum eStage : uint8_t {
            BG,
            POST
        };
        explicit CElement(eStage s) : m_stage(s) {}

        std::vector<UP<IPassElement>> draw() override {
            const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
            const auto CUR = g_pHyprRenderer->m_renderData.currentFB;
            if (!MON || !CUR || CUR->m_size.x <= 0)
                return {};
            auto& st = g_mon[MON->m_id];

            if (m_stage == BG) {
                const bool FRESH = !st.bg || st.bg->m_size != CUR->m_size || st.bg->m_drmFormat != CUR->m_drmFormat;
                if (!ensureFB(st.bg, CUR, "hyprhole-bg"))
                    return {};
                if (FRESH)
                    g_pHyprRenderer->damageMonitor(MON);
                blitDamage(MON, CUR, st.bg);
                return {};
            }

            const bool FRESH = !st.last || st.last->m_size != CUR->m_size || st.last->m_drmFormat != CUR->m_drmFormat;
            if (!ensureFB(st.last, CUR, "hyprhole-last"))
                return {};
            if (FRESH) {
                g_pHyprRenderer->damageMonitor(MON);
                st.lastValid = false;
            }
            // pictures of what is leaving, from the frame before
            for (auto& f : st.fx)
                if (f->needSnap) {
                    f->needSnap   = false;
                    const CBox& B = f->snapBox;
                    const int   X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
                    const int   X1 = std::min((int)CUR->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)CUR->m_size.y, (int)(B.y + B.h));
                    if (X1 - X0 < 4 || Y1 - Y0 < 4 || !ensureFB(f->snap, X1 - X0, Y1 - Y0, CUR->m_drmFormat, "hyprhole-snap")) {
                        f->dead = true;
                        continue;
                    }
                    blitRect(st.last, f->snap, X0, Y0, X1, Y1, 0, 0);
                }
            blitDamage(MON, CUR, st.last); // now: this frame, as Hyprland drew it
            if (!FRESH)
                st.lastValid = true;

            const auto DW   = movedWindow(MON);
            SEffect*   drag = nullptr;
            for (auto& f : st.fx)
                if (f->kind == DRAG && !f->dead)
                    drag = f.get();
            if (DW && !drag && st.lastValid)
                startDrag(st, MON, DW, CUR);
            else if (drag && !drag->released && (!DW || (uintptr_t)DW.get() != drag->id))
                releaseDrag(*drag, MON);

            if (st.fx.empty() || !ensureGL())
                return {};
            for (auto& f : st.fx)
                if (!f->dead && !step(st, *f, MON))
                    f->dead = true;
            for (auto& f : st.fx)
                if (f->dead && f->hidden)
                    showWindow(*f);
            std::erase_if(st.fx, [](const auto& f) { return f->dead; });
            if (st.fx.empty())
                return {};

            const auto TARGET = fbID(CUR);
            if (!TARGET || !st.bg || !ensureFB(st.scene, CUR, "hyprhole-scene"))
                return {};
            SGLState gs;
            gs.save();
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);

            // the real windows under live effects give way to the background
            for (auto& f : st.fx)
                if (!f->snap) {
                    const CBox& B  = f->live;
                    const int   X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
                    const int   X1 = std::min((int)CUR->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)CUR->m_size.y, (int)(B.y + B.h));
                    blitRect(st.bg, CUR, X0, Y0, X1, Y1, X0, Y0);
                }

            glBindFramebuffer(GL_FRAMEBUFFER, *TARGET);
            glViewport(0, 0, (GLsizei)CUR->m_size.x, (GLsizei)CUR->m_size.y);
            glBindVertexArray(g_gl.vao);
            // the desktop first, then windows, the dragged one on top
            for (int pass = 0; pass < 3; ++pass)
                for (auto& f : st.fx) {
                    const int P = f->kind == WORKSPACE ? 0 : f->kind == DRAG ? 2 : 1;
                    if (P == pass)
                        drawFx(st, *f, CUR, true);
                }
            gs.restore();
            return {};
        }

        bool needsLiveBlur() override {
            return false;
        }
        bool needsPrecomputeBlur() override {
            return false;
        }
        const char* passName() override {
            return "CHyprholeElement";
        }
        ePassElementType type() override {
            return EK_CUSTOM;
        }
        std::optional<CBox> boundingBox() override {
            return std::nullopt;
        }
        bool disableSimplification() override {
            return true;
        }
        bool undiscardable() override {
            return true;
        }

      private:
        eStage m_stage;
    };

    void onRenderStage(eRenderStage stage) {
        if (stage == RENDER_POST) {
            // keep frames coming (with full damage) while anything runs
            for (const auto& mon : State::monitorState()->monitors())
                if (mon && g_mon.contains(mon->m_id) && (!g_mon[mon->m_id].fx.empty() || movedWindow(mon)))
                    g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        if (stage != RENDER_PRE_WINDOWS && stage != RENDER_POST_WINDOWS)
            return;
        if (!ownMonitorFrame())
            return;
        const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
        if (!MON)
            return;
        if (stage == RENDER_PRE_WINDOWS) {
            watch(MON);
            g_pHyprRenderer->m_renderPass.add(makeUnique<CElement>(CElement::BG));
        } else
            g_pHyprRenderer->m_renderPass.add(makeUnique<CElement>(CElement::POST));
    }
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    g_listeners.push_back(Event::bus()->m_events.render.stage.listen([](eRenderStage s) { onRenderStage(s); }));
    return {"hyprhole", "Every window change goes through a black hole", "guizzan", "0.1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_listeners.clear();
    for (auto& [id, st] : g_mon)
        for (auto& f : st.fx)
            if (f->hidden)
                showWindow(*f); // never leave a window hidden
    g_mon.clear();
    if (g_gl.prog)
        glDeleteProgram(g_gl.prog);
    if (g_gl.vao)
        glDeleteVertexArrays(1, &g_gl.vao);
    g_gl = {};
}
