import math, time, sys, subprocess, array

def load_image(path):
    """Load an image via ffmpeg, returning (w, h, a).

    `a` is an array.array('B') of w*h*3 bytes in RGB order.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "image2pipe",
         "-pix_fmt", "rgb24", "-vcodec", "ppm", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + err.decode("utf-8", "replace"))

    # Parse the PPM header: "P6\n<w> <h>\n<maxval>\n" followed by raw bytes.
    # Whitespace between fields may be any mix of spaces/newlines, and
    # comment lines starting with '#' are allowed.
    if out[:2] != b"P6":
        raise ValueError("unexpected ffmpeg output (not a P6 PPM)")

    fields = []          # collected header integers: width, height, maxval
    i = 2
    n = len(out)
    while len(fields) < 3:
        # skip whitespace
        while i < n and out[i] in b" \t\r\n":
            i += 1
        # skip comments
        if i < n and out[i:i + 1] == b"#":
            while i < n and out[i] not in b"\r\n":
                i += 1
            continue
        start = i
        while i < n and out[i] not in b" \t\r\n":
            i += 1
        fields.append(int(out[start:i]))
    # single whitespace char separates header from pixel data
    i += 1

    w, h, maxval = fields
    if maxval != 255:
        raise ValueError("unsupported maxval %d" % maxval)

    a = array.array("B", out[i:i + w * h * 3])
    if len(a) != w * h * 3:
        raise ValueError("truncated pixel data")
    return w, h, a

w, h = 1920, 1080

txs = [load_image("3dswr.png"),
       load_image("pyodide_logo.png"),
       load_image("python_logo.png"),
       load_image("numpy_logo.png")]

img = array.array('B', [0]) * (w*h*3)
ZEROS = array.array('B', [0]) * (w*h*3)

def triangle(x0, y0, u0, v0,
             x1, y1, u1, v1,
             x2, y2, u2, v2,
             L, tx):
    delta = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    ux = ((u1-u0)*(y2-y0) - (u2-u0)*(y1-y0)) / delta
    uy = ((x1-x0)*(u2-u0) - (x2-x0)*(u1-u0)) / delta
    uk = u0 - x0*ux - y0*uy
    vx = ((v1-v0)*(y2-y0) - (v2-v0)*(y1-y0)) / delta
    vy = ((x1-x0)*(v2-v0) - (x2-x0)*(v1-v0)) / delta
    vk = v0 - x0*vx - y0*vy

    nx01, ny01 = y1-y0, x0-x1
    nx12, ny12 = y2-y1, x1-x2
    nx20, ny20 = y0-y2, x2-x0
    k01 = -(x0*nx01 + y0*ny01)
    k12 = -(x1*nx12 + y1*ny12)
    k20 = -(x2*nx20 + y2*ny20)


    tw, th, tpx = tx
    umask = tw - 1
    vmask = th - 1
    xa = max(0, int(min(x0, x1, x2)))
    ya = max(0, int(min(y0, y1, y2)))
    xb = min(w, int(max(x0, x1, x2))+1)
    yb = min(h, int(max(y0, y1, y2))+1)

    for y in range(ya, yb):
        yf = y + 0.5
        xf = xa + 0.5
        e01 = xf*nx01 + yf*ny01 + k01
        e12 = xf*nx12 + yf*ny12 + k12
        e20 = xf*nx20 + yf*ny20 + k20
        u = xf*ux + yf*uy + uk
        v = xf*vx + yf*vy + vk
        wp = (y*w + xa) * 3
        for x in range(xa, xb):
            if e01 <= 0 and e12 <= 0 and e20 <= 0:
                ui = int(u) & umask
                vi = int(v) & vmask
                rp = (vi*tx[0] + ui)*3
                img[wp] = int(tpx[rp] * L)
                img[wp+1] = int(tpx[rp+1] * L)
                img[wp+2] = int(tpx[rp+2] * L)
            e01 += nx01
            e12 += nx12
            e20 += nx20
            u += ux
            v += vx
            wp += 3

N = 7
M = 13
R1 = 1.5
R2 = 0.5

circle = [(R1 + R2*math.cos(i*2*math.pi/N), R2*math.sin(i*2*math.pi/N), 0)
          for i in range(N)]

vertices = [(x*math.cos(i*2*math.pi/M)+z*math.sin(i*2*math.pi/M),
             y,
             z*math.cos(i*2*math.pi/M)-x*math.sin(i*2*math.pi/M))
            for i in range(M)
            for (x, y, z) in circle]

faces = [(txs[(i*M+j) % len(txs)], (i+1)%N+j*N, (i+1)%N+(j+1)%M*N, i+(j+1)%M*N, i+j*N)
         for j in range(M)
         for i in range(N)]

def render_quads(pts, faces, ra, rb, rc):
    xpts = []
    for i, (x, y, z) in enumerate(pts):
        x, y = x*math.cos(ra)+y*math.sin(ra), y*math.cos(ra)-x*math.sin(ra)
        x, z = x*math.cos(rb)+z*math.sin(rb), z*math.cos(rb)-x*math.sin(rb)
        y, z = y*math.cos(rc)+z*math.sin(rc), z*math.cos(rc)-y*math.sin(rc)
        xpts.append((w/2 + x*w/(8+z), h/2 - y*w/(8+z), x, y, z))
    for (tx, a, b, c, d) in sorted(faces, key = lambda f:-xpts[f[1]][-1]):
        xpa, xpb, xpc, xpd = xpts[a], xpts[b], xpts[c], xpts[d]
        if (xpb[0]-xpc[0])*(xpa[1]-xpc[1]) - (xpb[1]-xpc[1])*(xpa[0]-xpc[0]) > 0:
            ax, ay, az = xpa[2:]
            bx, by, bz = xpb[2:]
            cx, cy, cz = xpc[2:]
            nx = (cy-ay)*(bz-az) - (cz-az)*(by-ay)
            ny = (cz-az)*(bx-ax) - (cx-ax)*(bz-az)
            nz = (cx-ax)*(by-ay) - (cy-ay)*(bx-ax)
            nl = (nx**2 + ny**2 + nz**2) ** 0.5
            L = 0.5 + 0.49*abs(nz / nl)
            triangle(xpb[0], xpb[1], 512, 256,
                     xpa[0], xpa[1], 0, 256,
                     xpc[0], xpc[1], 512, 0,
                     L, tx)
            triangle(xpc[0], xpc[1], 512, 0,
                     xpa[0], xpa[1], 0, 256,
                     xpd[0], xpd[1], 0, 0,
                     L, tx)

def animate():
    print("Starting ffplay...", file=sys.stderr)
    ffplay = subprocess.Popen(
        ['ffplay', '-loglevel', 'error', '-autoexit',
         '-f', 'image2pipe', '-vcodec', 'ppm', '-i', '-'],
        stdin=subprocess.PIPE)
    header = ('P6\n%d %d\n255\n' % (w, h)).encode()

    frame_num = 0
    ra = rb = rc = 0
    t = 0
    last = time.time()
    t0 = time.time()
    sa = 2 * 0.21
    sb = 2 * 0.2345
    sc = 2 * 0.2456

    tt = 0
    try:
        while ffplay.poll() is None:
            frame_num += 1
            tt += 0.03
            img[:] = ZEROS
            t -= time.time()
            render_quads(vertices, faces, sa*tt, sb*tt, sc*tt)
            t += time.time()
            try:
                ffplay.stdin.write(header)
                ffplay.stdin.write(img.tobytes())
                ffplay.stdin.flush()
            except (BrokenPipeError, OSError):
                break
            delta = time.time() - last
            if delta > 1:
                print("%0.3f ms (%0.3f FPS)" %
                      (t*1000/delta/frame_num, frame_num/delta),
                      file=sys.stderr)
                t = frame_num = 0
                last = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ffplay.stdin.close()
        except Exception:
            pass
        ffplay.wait()

animate()
