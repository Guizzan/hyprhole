# hyprhole — black hole for Hyprland

Windows that collapse into black holes. A Hyprland plugin where every window is a mass. Close one and it collapses under its own gravity into a black hole that is never drawn: you only see what it does to space around it - real gravitational lensing, an Einstein ring of your desktop, a frame-dragging whirlpool and a gravitational-wave ripple.

**Demo and details: https://guizzan.github.io/hyprhole/**

https://github.com/Guizzan/hyprhole/raw/master/docs/demo.mp4

## What it does

- **Close** — The window collapses faster and faster, spins up, heats and sheds its outer layers into spiral streams; when it is small enough an event horizon forms with a ripple through space, then the hole evaporates. The other windows wait, then glide into the space.
- **Open** — The same played backwards: a black hole explodes into the window.
- **Move / resize / fullscreen** — The window carries mass: the desktop bends around it as it travels and a faint ripple spreads from it.
- **Workspace switch** — Every window of the old desktop collapses into its own drifting black hole; the new desktop's windows burst out of theirs.
- **Drag** — SUPER + drag collapses the window into a small black hole that follows the cursor on a spring; drop it and the window bursts back out in its new place.
- **Physics-inspired optics** — Point-mass lens equation (beta = theta - thetaE^2/theta) with slightly different Einstein radii per colour, frame dragging, tidal streams, a chromatic gravitational-wave ripple.

## Install

Built and tested against **Hyprland 0.56** (Lua config). Plugins are tied to the exact Hyprland version — rebuild after every update.

```sh
make            # builds hyprhole-dev.so
make install    # copies it to hyprhole.so with a new inode (safe while Hyprland runs)
```

In your Lua config:

```lua
pcall(hl.plugin.load, os.getenv("HOME") .. "/path/to/hyprhole/hyprhole.so")
-- the plugin replaces Hyprland's own window and workspace animations
for _, leaf in ipairs({ "workspaces", "windowsIn", "windowsOut", "windowsMove", "fadeIn", "fadeOut" }) do
    hl.animation({ leaf = leaf, enabled = false })
end
```

Then log out and back in. Never hot-swap the plugin in a running session (unloading it can crash Hyprland), and run only one of these effect plugins at a time.

Try it in a nested session first: `Hyprland -c test/hyprland.lua`.

## See also

[hyprmorph](https://github.com/Guizzan/hyprmorph) (pixel morph) · [hyprliquid](https://github.com/Guizzan/hyprliquid) (liquid glass)
