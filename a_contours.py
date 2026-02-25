import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
import os


def heaviside_epsilon(z, epsilon=1.0):
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(z / epsilon))


def compute_averages(u0, phi, epsilon=1.0):
    H = heaviside_epsilon(phi, epsilon)
    num1 = np.sum(u0 * H)
    den1 = np.sum(H)
    c1 = num1 / den1 if den1 > 1e-10 else np.mean(u0)

    num2 = np.sum(u0 * (1.0 - H))
    den2 = np.sum(1.0 - H)
    c2 = num2 / den2 if den2 > 1e-10 else np.mean(u0)
    return c1, c2


def curvature_central(phi):
    fy, fx = np.gradient(phi)
    mag = np.sqrt(fx**2 + fy**2 + 1e-8)
    nx = fx / mag
    ny = fy / mag
    _, nxx = np.gradient(nx)
    nyy, _ = np.gradient(ny)
    return nxx + nyy


def chan_vese(u0, phi0, mu=0.25, nu=0, lambda1=1.0, lambda2=1.0, dt=1.0, epsilon=1.0, max_iter=300, tol=1e-4):
    phi = phi0.astype(np.float64).copy()
    phi = phi / (np.abs(phi).max() + 1e-10)
    history = [phi.copy()]

    for it in range(max_iter):
        phi_old = phi.copy()
        c1, c2 = compute_averages(u0, phi, epsilon)
        kappa = curvature_central(phi)
        data = -lambda1 * (u0 - c1)**2 + lambda2 * (u0 - c2)**2
        dphi = dt * (mu * kappa - nu + data)
        max_step = np.abs(dphi).max()
        if max_step > 0.5:dphi = dphi * (0.5 / max_step)
        phi = phi + dphi
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
        phi = np.clip(phi, -4.0, 4.0)
        change = np.sqrt(np.mean((phi - phi_old)**2))
        if it % 10 == 0:
            history.append(phi.copy())
        if it % 50 == 0:
            print(f"  Iter {it:4d}: c1={c1:.4f}, c2={c2:.4f}, "f"rms={change:.6f}")
        if change < tol and it > 10:
            print(f"  Converged at iteration {it}")
            break

    history.append(phi.copy())
    seg = (phi >= 0).astype(np.uint8)
    return phi, c1, c2, seg, history

def init_circle(shape, center=None, radius=None):
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 3
    y, x = np.ogrid[:h, :w]
    return radius - np.sqrt((x - center[1])**2 + (y - center[0])**2)


def init_checkerboard(shape, n=5):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return np.sin(np.pi * n * x / w) * np.sin(np.pi * n * y / h)

def create_test_images():
    images = {}
    np.random.seed(42)
    y100, x100 = np.ogrid[:100, :100]

    img1 = np.ones((100, 100)) * 0.7
    img1[(x100 - 35)**2 + (y100 - 35)**2 < 15**2] = 0.3
    rect = (x100 > 55) & (x100 < 85) & (y100 > 55) & (y100 < 85)
    img1[rect] = 0.3
    hole = (x100 > 63) & (x100 < 77) & (y100 > 63) & (y100 < 77)
    img1[hole] = 0.7
    img1 += np.random.normal(0, 0.08, img1.shape)
    img1 = np.clip(img1, 0, 1)
    images['noisy_shapes'] = img1
    r = np.sqrt((x100 - 50)**2 + (y100 - 50)**2)
    img2 = np.exp(-r**2 / (2 * 20**2))
    images['smooth_blob'] = img2
    img3 = np.zeros((100, 100))
    img3[25:75, 25:75] = 1.0
    images['two_regions'] = img3
    img4 = np.zeros((100, 100))
    for cy, cx in [(30, 25), (30, 75), (70, 50)]:
        img4[(x100 - cx)**2 + (y100 - cy)**2 < 14**2] = 1.0
    images['three_disks'] = img4

    return images

def run_demo():
    print("=" * 60)
    print("CHAN-VESE ACTIVE CONTOURS WITHOUT EDGES")
    print("Implementation from: Chan & Vese, IEEE T-IP 2001")
    print("=" * 60)

    os.makedirs('output', exist_ok=True)
    images = create_test_images()

    params = {
        'noisy_shapes': dict(mu=0.1,  dt=1.0, epsilon=1.0, max_iter=400,
                             init='circle', radius_frac=2.5),
        'smooth_blob':  dict(mu=0.02, dt=1.0, epsilon=1.5, max_iter=500,
                             init='circle', radius_frac=2.5),
        'two_regions':  dict(mu=0.1,  dt=1.0, epsilon=1.0, max_iter=400,
                             init='circle', radius_frac=2.5),
        'three_disks':  dict(mu=0.1,  dt=1.0, epsilon=1.0, max_iter=500,
                             init='checker'),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, img) in enumerate(images.items()):
        print(f"\n--- {name} ---")
        p = params[name]

        if p['init'] == 'checker':
            phi0 = init_checkerboard(img.shape, n=5)
        else:
            phi0 = init_circle(img.shape,
                               radius=min(img.shape) // p['radius_frac'])

        phi, c1, c2, seg, hist = chan_vese(
            img, phi0,
            mu=p['mu'], dt=p['dt'], epsilon=p['epsilon'],
            max_iter=p['max_iter'],
        )
        print(f"  Final c1={c1:.4f}, c2={c2:.4f}")

        ax = axes[0, idx]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.contour(phi, levels=[0], colors='r', linewidths=2)
        ax.set_title(f'{name}\nOriginal + Contour')
        ax.axis('off')

        ax = axes[1, idx]
        ax.imshow(seg, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Segmentation\n$\\mu$={p["mu"]}, c1={c1:.2f}, c2={c2:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output/chan_vese_demo.png', dpi=150, bbox_inches='tight')
    print("\nSaved output/chan_vese_demo.png")

    print("\n" + "=" * 60)
    print("Generating evolution visualisation...")
    print("=" * 60)

    img = images['noisy_shapes']
    phi0 = init_circle(img.shape, radius=40)
    phi, c1, c2, seg, history = chan_vese(
        img, phi0, mu=0.1, dt=1.0, epsilon=1.0, max_iter=400,
    )

    n_panels = 8
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    axes2 = axes2.flatten()
    step = max(1, (len(history) - 1) // (n_panels - 1))
    for i, ax in enumerate(axes2):
        hi = min(i * step, len(history) - 1)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.contour(history[hi], levels=[0], colors='r', linewidths=2)
        it_label = hi * 10 if hi < len(history) - 1 else 'Final'
        ax.set_title(f'Iteration {it_label}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output/chan_vese_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved output/chan_vese_evolution.png")
    plt.show()


if __name__ == "__main__":
    run_demo()
