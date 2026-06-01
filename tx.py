import numpy as np
from PIL import Image
import math, time, pygame

w, h = 1920, 1080

txs = [np.uint8(Image.open("3dswr.png")).copy(),
       np.uint8(Image.open("pyodide_logo.png")).copy(),
       np.uint8(Image.open("python_logo.png")).copy(),
       np.uint8(Image.open("numpy_logo.png")).copy()]

X = np.tile(np.array(range(w), dtype=np.int32), (h, 1))
Y = np.broadcast_to(np.array(range(h), dtype=np.int32).reshape(h, 1), (h, w))
aa = Y*w + X
X = np.float32(X + 0.5)
Y = np.float32(Y + 0.5)

img = np.zeros((h, w, 3), np.uint8)

def triangle(x0, y0, u0, v0,
             x1, y1, u1, v1,
             x2, y2, u2, v2,
             L, tx):
    # ux (x1-x0) + uy (y1-y0) = u1-u0
    # ux (x2-x0) + uy (y2-y0) = u2-u0
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

    th, tw = tx.shape[:2]
    xa = max(0, int(min(x0, x1, x2)))
    ya = max(0, int(min(y0, y1, y2)))
    xb = min(w, int(max(x0, x1, x2))+1)
    yb = min(h, int(max(y0, y1, y2))+1)
    Xb = X[ya:yb, xa:xb]
    Yb = Y[ya:yb, xa:xb]
    aab = aa[ya:yb, xa:xb]
    mask = (np.uint8(Xb*nx01 + Yb*ny01 + k01 <= 0) &
            np.uint8(Xb*nx12 + Yb*ny12 + k12 <= 0) &
            np.uint8(Xb*nx20 + Yb*ny20 + k20 <= 0))
    xya = aab[mask == 1]
    xx = X.ravel()[xya]
    yy = Y.ravel()[xya]
    uv = np.int32(xx*vx + yy*vy + vk) & (th-1)
    uv *= tw
    uv += np.int32(xx*ux + yy*uy + uk) & (tw-1)
    img.reshape((w*h, 3))[xya] = np.int32(tx.reshape((tw*th,3))[uv] * L)

def pure_python_triangle(x0, y0, u0, v0,
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

    th, tw = tx.shape[:2]
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
        for x in range(xa, xb):
            if e01 <= 0 and e12 <= 0 and e20 <= 0:
                ui = int(u) & umask
                vi = int(v) & vmask
                px = tx[vi, ui]
                img[y, x, 0] = int(px[0] * L)
                img[y, x, 1] = int(px[1] * L)
                img[y, x, 2] = int(px[2] * L)
            e01 += nx01
            e12 += nx12
            e20 += nx20
            u += ux
            v += vx

triangle_fn = triangle

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
            triangle_fn(xpb[0], xpb[1], 512, 256,
                        xpa[0], xpa[1], 0, 256,
                        xpc[0], xpc[1], 512, 0,
                        L, tx)
            triangle_fn(xpc[0], xpc[1], 512, 0,
                        xpa[0], xpa[1], 0, 256,
                        xpd[0], xpd[1], 0, 0,
                        L, tx)

def animate_with_pygame():
    pygame.init()
    global pxcount, tricount, triangle_fn
    height, width = img.shape[:2]
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption('Live Animation')

    clock = pygame.time.Clock()
    running = True

    frame_num = 0
    ra = rb = rc = 0
    t = 0
    last = time.time()
    t0 = time.time()
    sa = 2 * 0.21
    sb = 2 * 0.2345
    sc = 2 * 0.2456

    tt = 0
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_p:
                    triangle_fn = pure_python_triangle if triangle_fn is triangle else triangle
                    print("rasterizer:", triangle_fn.__name__)
                    t = frame_num = 0
                    last = time.time()
        frame_num += 1
        tt += 0.03#time.time() - t0
        img[:,:,:] = 0
        pxcount = tricount = 0
        t -= time.time()
        render_quads(vertices, faces, sa*tt, sb*tt, sc*tt)
        t += time.time()

        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        delta = time.time() - last
        if delta > 1:
            print("%0.3f ms (%0.3f FPS)" %
                  (t*1000/delta/frame_num, frame_num/delta))
            t = frame_num = 0
            last = time.time()

    pygame.quit()

animate_with_pygame()
