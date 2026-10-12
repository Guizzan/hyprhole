// hyprhole — windows that collapse into black holes.
//
// A window is a mass. Close it and it collapses under its own gravity: it
// shrinks faster and faster (spinning up as it contracts, heating to a
// blue-white glow), space around it bends more and more as it gets denser,
// its outer layers peel off into spiral streams, and when it is small enough
// an event horizon forms with a gravitational-wave ripple: the window has
// become a black hole. Nothing is drawn for the hole: it shows only through
// what it does to space - the desktop around it folds into a mirrored
// Einstein ring with rainbow edges and is wound into a whirlpool by the
// spinning mass. Then the hole evaporates and space relaxes.
//
//   * opening: the same played backwards: a black hole explodes into a window;
//   * moving / resizing / fullscreen / layout changes: the window's mass bends
//     the light around it as it travels, with colour fringes and a ripple;
//   * workspace switch: every window of the old desktop collapses into its own
//     black hole drifting away, the new desktop's windows burst out of theirs;
//   * dragging (SUPER + mouse): the window collapses into a small black hole
//     that follows the cursor; dropping it makes it burst back out.
//
// Optics: background light is bent by the point-mass lens equation
// (beta = theta - thetaE^2 / theta), giving the Einstein ring and the mirrored
// image around the horizon; red, green and blue use slightly different
// Einstein radii (chromatic fringing); frame dragging twists it; ripples
// displace each colour slightly differently.
//
// Rendering: every frame the damaged part of the composited windows is kept
// in `last` and of the background (before windows) in `bg`. While effects
// run, the real windows under them are painted over with the background and
// each effect is drawn from `last` (live content) or a snapshot (closing
// windows, the dragged window). Hyprland's own window and workspace
// animations must be disabled.

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
    constexpr float T_CLOSE = 0.70f;
    constexpr float T_OPEN  = 0.70f;
    constexpr float T_MOVE  = 0.45f;
    constexpr float T_WS_CLOSE = 0.70f; // workspace switch: old windows collapse
    constexpr float T_WS_OPEN  = 0.62f; //                   new windows burst out
    constexpr float COLLAPSE   = 0.55f; // share of a close before the horizon forms
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
        float rotA   = 0.f;    // spin of the collapsing window
        float heat   = 0.f;    // blue-white glow of the compressed window
        float shed   = 0.f;    // outer layers peeling off into streams
        float sGhost = 1.f, twistGhost = 0.f;
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
        V2                       drift; // workspace switch: where the hole drifts (px)
        float                    delay = 0.f; // MOVE: wait (at the old place, old picture) for a close next to it
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
uniform float rotA;
uniform float heat;
uniform float shed;
uniform float sGhost;
uniform float twistGhost;
uniform float winAlpha;
uniform float spin;          // +1 / -1: which way it turns
uniform float reach;         // beyond this the effect fades out (px)
uniform vec2  vel;
uniform float time;
uniform float px;            // 1 logical px
uniform vec2  fbSize;
uniform int   occN;          // windows in front of this effect (a closing window lies behind the others)
uniform vec4  occ[16];
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

vec4 sampleWin(vec2 src, float r1, float rimPx, out float sdOut) {
    vec2  uv = (src - winBox.xy) / winBox.zw;
    float sd = sdRound(src - winBox.xy - winBox.zw * 0.5, winBox.zw * 0.5, corner);
    sdOut    = sd;
    float m  = 1.0 - smoothstep(-rimPx, rimPx, sd);
    if (uv.x < -0.01 || uv.y < -0.01 || uv.x > 1.01 || uv.y > 1.01)
        m = 0.0;
    vec3 col = texture(content, contentUv.xy + clamp(uv, 0.0, 1.0) * contentUv.zw).rgb;
    if (rh > 0.0) {
        // light climbing out near the horizon is redshifted and dimmed
        float g = smoothstep(rh, rh * 3.0, r1);
        col = mix(col * vec3(1.0, 0.35, 0.18) * 0.6, col, g);
        m *= smoothstep(rh * 0.98, rh * 1.25, r1);
    }
    return vec4(col, m);
}

// The collapsing window itself: shrunk about the centre by `s`, spun by rotA,
// glowing as it heats. `glow` returns the light it radiates around itself.
vec4 windowCore(vec2 q, out float glow) {
    vec2  rel = rot(q - c, -spin * rotA);
    float ss  = max(s, 1e-3);
    vec2  src = c + rel / ss;
    float sd;
    vec4  w = sampleWin(src, length(q - c), 0.8 / ss, sd);
    float screenSd = sd * ss;
    glow = heat * exp(-max(screenSd, 0.0) / (14.0 * px)) * step(0.0, screenSd);
    w.rgb = mix(w.rgb, vec3(0.80, 0.88, 1.0) * 1.5, heat * 0.8);
    w.a  *= winAlpha;
    return w;
}

// Its outer layers, peeling off: they lag behind the core and are wound into
// spiral streams (inner parts orbit faster).
vec4 windowShed(vec2 q) {
    vec2  rel = q - c;
    float r1  = length(rel);
    float L   = L0();
    float th  = atan(rel.y, rel.x) - spin * twistGhost / (r1 / L + 0.25);
    float ss  = max(sGhost, 1e-3);
    float r0  = r1 / ss;
    for (int i = 0; i < 3; i++)
        r0 = r1 / pow(ss, 1.0 + 2.5 * L * 0.2 / (r0 + L * 0.2));
    float sd;
    vec4 w = sampleWin(c + vec2(cos(th), sin(th)) * r0, r1, 1.0, sd);
    return w;
}

// The bent position of the light arriving at screen point p, for Einstein
// radius e. Nothing is drawn for the hole itself: it only shows in what it
// does to space - the desktop folds into a mirrored Einstein ring around it
// and is wound into a whirlpool by the spinning mass (frame dragging).
vec2 lens(vec2 p, float e) {
    vec2  d = p - c;
    float r = max(length(d), 1e-3);
    float fall = 1.0 - smoothstep(reach * 0.55, reach, r);
    float b = r - e * e / r * fall;
    float R = max(max(rh, e * 0.45), 1.0);
    vec2 dir = rot(d / r, spin * twist * R * R / (r * r + R * R) * fall);
    return c + dir * b;
}

// a gravitational wave passing: space is pushed in and out, each colour by
// slightly different amounts (a chromatic ripple, no light of its own)
vec2 ripple(vec2 p, float shift) {
    vec2  d = p - c;
    float r = length(d);
    if (flashA <= 0.0 || r < 1e-3)
        return p;
    float k = (r - flashR) / (16.0 * px) + shift;
    return p + d / r * flashA * 11.0 * px * sin(k * 2.2) * exp(-k * k * 0.15);
}

void main() {
    vec2  p0 = gl_FragCoord.xy;
    for (int i = 0; i < 16; i++) {
        if (i >= occN)
            break;
        vec4 o = occ[i];
        if (p0.x >= o.x && p0.y >= o.y && p0.x < o.x + o.z && p0.y < o.y + o.w) {
            frag = texture(scene, p0 / fbSize); // that window is in front: leave it as it is
            return;
        }
    }
    vec2  d  = p0 - c;
    float r  = length(d);
    float fade = 1.0 - smoothstep(reach * 0.85, reach, r);
    vec2  pr = ripple(p0, -0.35), pg = ripple(p0, 0.0), pb = ripple(p0, 0.35);

    // space behind, bent by the window's mass, each colour a little differently
    vec3 col = vec3(texture(scene, lens(pr, E * 1.05) / fbSize).r, texture(scene, lens(pg, E) / fbSize).g, texture(scene, lens(pb, E * 0.95) / fbSize).b);

    // the shed layers, then the window itself (colour fringes along its motion)
    if (shed > 0.001) {
        vec4 g = windowShed(pg);
        col = mix(col, g.rgb, g.a * shed * 0.55);
    }
    vec2  f = vel * 0.012 + (pg - c) * 0.012 * heat; // + a little dispersion as it heats
    float gr, gg, gb;
    vec4  wr = windowCore(pr - f, gr), wg = windowCore(pg, gg), wb = windowCore(pb + f, gb);
    col = vec3(mix(col.r, wr.r, wr.a), mix(col.g, wg.g, wg.a), mix(col.b, wb.b, wb.a));
    col += vec3(gr * 0.6, gg * 0.7, gb * 0.85) * vec3(0.75, 0.85, 1.0) * winAlpha * 0.7;

    // the shadow: no drawn disk, just light failing to get out near the centre
    if (rh > 0.0)
        col *= smoothstep(rh * 0.35, rh * 1.1, r);

    vec3 base = texture(scene, gl_FragCoord.xy / fbSize).rgb;
    frag = vec4(mix(base, col, fade), 1.0);
}
)";

    struct SGL {
        GLuint prog = 0, vao = 0;
        GLint  uScene = -1, uContent = -1, uContentUv = -1, uWin = -1, uCorner = -1, uC = -1, uS = -1, uRh = -1, uE = -1, uTwist = -1, uDisk = -1, uFlashR = -1,
              uFlashA = -1, uOccN = -1, uOcc = -1, uRotA = -1, uHeat = -1, uShed = -1, uSGhost = -1, uTwistGhost = -1, uWinAlpha = -1, uSpin = -1, uReach = -1, uVel = -1, uTime = -1, uPx = -1, uFb = -1;
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
        g_gl.uRotA      = U("rotA");
        g_gl.uOccN      = U("occN");
        g_gl.uOcc       = U("occ[0]");
        g_gl.uHeat      = U("heat");
        g_gl.uShed      = U("shed");
        g_gl.uSGhost    = U("sGhost");
        g_gl.uTwistGhost = U("twistGhost");
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

    // A window collapsing into a black hole, at time t (s) of a collapse whose
    // horizon forms at tc. `evaporate`: the hole fades away afterwards.
    void collapse(SParams& p, const SMonitorState& st, float t, float tc, float rhPeak, bool evaporate) {
        const float MIND = (float)std::min(p.win.w, p.win.h), MAXD = (float)std::max(p.win.w, p.win.h);
        const float K    = std::clamp(t / tc, 0.f, 1.f);
        const float SEND = std::clamp(rhPeak * 0.9f / std::max(MIND * 0.5f, 1.f), 0.02f, 0.6f);
        p.s       = 1.f - (1.f - SEND) * std::pow(K, 2.2f); // gravity wins faster and faster
        p.rotA    = std::min(0.12f * (1.f / p.s - 1.f), 4.f); // spins up as it contracts
        p.heat    = smooth01((K - 0.3f) / 0.7f);
        p.corner  = (float)(WIN_CORNER * st.scale) + (MIND * 0.5f - (float)(WIN_CORNER * st.scale)) * smooth01(std::pow(K, 1.3f));
        p.shed    = smooth01((K - 0.25f) / 0.3f) * (1.f - smooth01((t - tc) / 0.25f));
        p.sGhost  = std::pow(p.s, 0.55f);
        p.twistGhost = 2.4f * K;
        const float FORMED = smooth01((t - tc) / 0.06f);
        const float LEFT   = evaporate ? 1.f - smooth01((t - (tc + 0.25f)) / 0.18f) : 1.f;
        p.rh      = t < tc ? 0.f : rhPeak * FORMED * LEFT;
        // the lensing grows as the mass gets compact, then is the hole's
        p.E       = t < tc ? EINSTEIN * rhPeak * std::pow(smooth01((K - 0.05f) / 0.95f), 1.5f) : EINSTEIN * p.rh;
        p.alpha   = 1.f - smooth01((t - tc) / 0.05f); // the core is gone behind its horizon
        p.twist   = 3.2f * (t < tc ? K * K : LEFT); // the desktop is wound into a whirlpool
        if (t >= tc && t < tc + 0.35f) {
            const float F = (t - tc) / 0.35f;
            p.flashR      = F * MAXD * 1.1f;
            p.flashA      = 1.f - F;
        }
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
        // a close is COLLAPSE of the run until the horizon, the rest the hole
        const auto  RUN = [&](float tt) { return tt * (COLLAPSE + 0.45f) / 1.f; };

        switch (f.kind) {
            case CLOSE: {
                const float DR = smooth01(t);
                p.win = CBox{f.from.x + f.drift.x * DR, f.from.y + f.drift.y * DR, f.from.w, f.from.h};
                p.c   = centreOf(p.win);
                collapse(p, st, RUN(t), COLLAPSE, f.rhPeak, true);
                break;
            }
            case OPEN: {
                // a close played backwards: a black hole explodes into the window
                const float DR = 1.f - smooth01(t);
                p.win = CBox{f.to.x + f.drift.x * DR, f.to.y + f.drift.y * DR, f.to.w, f.to.h};
                p.c   = centreOf(p.win);
                collapse(p, st, RUN(1.f - t), COLLAPSE, f.rhPeak, true);
                break;
            }
            case MOVE: {
                if (T < f.delay) { // a neighbour is still collapsing: wait in place
                    p.win   = f.from;
                    p.c     = centreOf(p.win);
                    f.prevC = p.c;
                    break;
                }
                const float t = std::clamp((T - f.delay) / f.dur, 0.f, 1.f);
                const float K = easeInOut(t);
                p.win = lerpBox(f.from, f.to, K);
                p.c   = centreOf(p.win);
                const float B = std::sin(3.14159265f * t);
                p.E      = 0.42f * (float)std::min(p.win.w, p.win.h) * 0.5f * B;
                p.flashR = t * (float)std::max(p.win.w, p.win.h) * 1.1f;
                p.flashA = (1.f - t) * 0.55f;
                const V2 C = p.c;
                p.vel      = DT > 0.f ? (C - f.prevC) * (1.f / DT) : V2{};
                f.prevC    = C;
                break;
            }
            case DRAG: {
                const V2    CUR = cursorGL(mon);
                const float RH  = (float)(DRAG_HOLE * S);
                if (!f.released) {
                    // the hole follows the cursor on a spring (it has mass)
                    const V2 A = (CUR - f.hole) * 600.f - f.holeV * 38.f;
                    f.holeV    = f.holeV + A * DT;
                    f.hole     = f.hole + f.holeV * DT;
                    p.c   = f.hole;
                    p.win = CBox{f.from.x + (f.hole.x - f.grabOff.x), f.from.y + (f.hole.y - f.grabOff.y), f.from.w, f.from.h};
                    collapse(p, st, std::min(T, T_PICK + 0.4f), T_PICK, RH, false);
                    p.vel = f.holeV;
                } else {
                    // burst back out, flying into place
                    const float K = std::clamp(T / T_DROP, 0.f, 1.f);
                    const float M = easeInOut(std::min(1.f, K / 0.6f));
                    p.c   = f.hole + (centreOf(f.to) - f.hole) * M;
                    p.win = CBox{p.c.x - f.to.w * 0.5, p.c.y - f.to.h * 0.5, f.to.w, f.to.h};
                    collapse(p, st, (1.f - K) * (T_PICK + 0.2f), T_PICK, RH, false);
                    if (T >= T_DROP)
                        return false;
                }
                break;
            }
        }
        return f.kind == DRAG || T < f.delay + f.dur;
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

    void addMove(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w, const CBox& fromL, float delay) {
        const CBox TO = glBox(w.box, mon);
        if (auto* f = findFx(st, w.id)) {
            if (f->kind == DRAG)
                return;
            if (f->kind == OPEN) {
                // a new window settling (layout, the app picking its size) in its
                // first moments: keep bursting out, just into the new place
                f->to   = TO;
                f->live = TO;
                return;
            }
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
        if (delay > 0.f) {
            // until the close beside it has played out it stays as it was
            f.delay    = delay;
            f.needSnap = true;
            f.snapBox  = f.from;
        }
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
            // workspace switch: the old desktop's windows collapse into black
            // holes drifting away, the new ones burst out of theirs
            const float DIR = WS->m_id > st.cachedWs ? 1.f : -1.f;
            const V2    DRIFT = {-DIR * (float)mon->m_pixelSize.x * 0.08f, 0.f};
            for (auto& f : st.fx)
                if (f->kind != DRAG)
                    f->dead = true;
            for (const auto& w : st.cached) {
                addClose(st, mon, w);
                st.fx.back()->dur   = T_WS_CLOSE;
                st.fx.back()->drift = DRIFT;
                st.fx.back()->spin  = DIR;
            }
            for (const auto& w : BOXES) {
                addOpen(st, mon, w);
                st.fx.back()->dur   = T_WS_OPEN;
                st.fx.back()->drift = DRIFT * -1.f;
                st.fx.back()->spin  = DIR;
            }
            st.cachedWs = WS->m_id;
            st.cached   = BOXES;
            g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        bool changed = false;
        // a window closing: the others wait until it has gone before filling its place
        bool closing = false;
        for (const auto& w : st.cached)
            if (!findBox(BOXES, w.id) && !(DW && w.id == (uintptr_t)DW.get()))
                closing = true;
        for (const auto& w : BOXES) {
            const auto OLD = findBox(st.cached, w.id);
            if (!OLD) {
                if (auto* f = findFx(st, w.id); f && f->kind == DRAG)
                    continue; // a dropped window: its drag effect lets it out
                addOpen(st, mon, w);
                changed = true;
            } else if (!sameBox(OLD->box, w.box)) {
                addMove(st, mon, w, OLD->box, closing ? T_CLOSE * 0.92f : 0.f);
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

    void drawFx(SMonitorState& st, SEffect& f, const SP<Render::IFramebuffer>& target, const PHLMONITOR& mon) {
        const bool  SNAP    = f.snap && !(f.kind == MOVE && secsSince(f.start) >= f.delay);
        const auto& CONTENT = SNAP ? f.snap : st.last;
        if (!CONTENT)
            return;
        const float FW = (float)target->m_size.x, FH = (float)target->m_size.y;
        const auto& p  = f.p;
        // how far the effect reaches: the window, the disk, the lensing, the ripple
        float reach = 0.f;
        for (const V2 K : {V2{0, 0}, V2{1, 0}, V2{0, 1}, V2{1, 1}})
            reach = std::max(reach, len(V2{(float)(p.win.x + p.win.w * K.x), (float)(p.win.y + p.win.h * K.y)} - p.c));
        reach = std::max({reach * 1.15f + 40.f * st.scale, p.rh * 9.f, p.E * 4.f, p.flashR + 60.f * st.scale});

        const int X0 = std::clamp((int)std::floor(p.c.x - reach), 0, (int)FW), X1 = std::clamp((int)std::ceil(p.c.x + reach), 0, (int)FW);
        const int Y0 = std::clamp((int)std::floor(p.c.y - reach), 0, (int)FH), Y1 = std::clamp((int)std::ceil(p.c.y + reach), 0, (int)FH);
        if (X1 <= X0 || Y1 <= Y0)
            return;
        // effects stack: each sees the ones drawn before it (only its own region is copied)
        blitRect(target, st.scene, X0, Y0, X1, Y1, X0, Y0);
        glBindFramebuffer(GL_FRAMEBUFFER, *fbID(target));

        // a closing window lies behind the others: windows (where they are drawn
        // now) cover its hole as they move into its place
        std::vector<float> occ;
        if (f.kind == CLOSE)
            for (const auto& w : st.cached) {
                if (occ.size() >= 16 * 4)
                    break;
                CBox B = glBox(w.box, mon);
                if (const auto* o = findFx(st, w.id)) {
                    if (o->kind == DRAG || (o->kind == OPEN && o->p.s < 0.6f))
                        continue;
                    B = o->p.win;
                }
                occ.insert(occ.end(), {(float)B.x, (float)B.y, (float)B.w, (float)B.h});
            }
        glActiveTexture(GL_TEXTURE0);
        linearClamp(st.scene);
        glActiveTexture(GL_TEXTURE1);
        linearClamp(CONTENT);

        glUseProgram(g_gl.prog);
        glUniform1i(g_gl.uScene, 0);
        glUniform1i(g_gl.uContent, 1);
        if (SNAP)
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
        glUniform1f(g_gl.uRotA, p.rotA);
        glUniform1f(g_gl.uHeat, p.heat);
        glUniform1f(g_gl.uShed, p.shed);
        glUniform1f(g_gl.uSGhost, p.sGhost);
        glUniform1f(g_gl.uTwistGhost, p.twistGhost);
        glUniform1f(g_gl.uWinAlpha, p.alpha);
        glUniform1f(g_gl.uSpin, f.spin);
        glUniform1f(g_gl.uReach, reach);
        glUniform2f(g_gl.uVel, p.vel.x, p.vel.y);
        glUniform1f(g_gl.uTime, secsSince(f.start));
        glUniform1f(g_gl.uPx, st.scale);
        glUniform2f(g_gl.uFb, FW, FH);

        glUniform1i(g_gl.uOccN, (GLint)(occ.size() / 4));
        if (!occ.empty())
            glUniform4fv(g_gl.uOcc, (GLsizei)(occ.size() / 4), occ.data());
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
                if (f->kind == OPEN || f->kind == MOVE) {
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
                    const int P = f->kind == CLOSE ? 0 : f->kind == DRAG ? 2 : 1;
                    if (P == pass)
                        drawFx(st, *f, CUR, MON);
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
