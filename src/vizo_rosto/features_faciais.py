import numpy as np, pandas as pd

def load(path):
    df = pd.read_csv(path, sep=';', index_col=False)
    if df.shape[1] == 1:                       # provavelmente separado por vírgula
        df = pd.read_csv(path, sep=',', index_col=False)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    if 'ct_face_x0' not in df.columns:
        ct = sum(c.startswith('ct_') for c in df.columns)
        raise SystemExit(
            f"O CSV '{path}' nao tem as colunas de contorno (ct_face_x0...). "
            f"{df.shape[1]} colunas, {ct} comecam com 'ct_'. Primeiras: {list(df.columns[:8])}"
        )
    return df

def pts(row, prefix, n):
    return np.array([[row[f'{prefix}_x{i}'], row[f'{prefix}_y{i}']] for i in range(n)], float)

def width_at(poly, y):
    """largura do contorno (polígono fechado) na altura y"""
    xs = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return (max(xs) - min(xs)) if len(xs) >= 2 else np.nan

def angle(a, b, c):
    """ângulo em b (graus) formado por a-b-c"""
    v1, v2 = a - b, c - b
    cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def extract(row):
    face = pts(row, 'ct_face', 36)
    lb = pts(row, 'ct_leftEyebrowBottom', 5); rb = pts(row, 'ct_rightEyebrowBottom', 5)
    lt = pts(row, 'ct_leftEyebrowTop', 5);    rt = pts(row, 'ct_rightEyebrowTop', 5)
    le = pts(row, 'ct_leftEye', 16); re = pts(row, 'ct_rightEye', 16)
    ul = pts(row, 'ct_upperLipTop', 11); ll = pts(row, 'ct_lowerLipBottom', 9)
    nb = pts(row, 'ct_noseBottom', 3)

    # 1) corrige a rotação (roll) alinhando a linha dos olhos na horizontal
    c_le, c_re = le.mean(0), re.mean(0)
    d = c_le - c_re
    th = np.arctan2(d[1], d[0])
    if abs(th) > np.pi/2: th -= np.sign(th)*np.pi          # evita inverter por causa de esquerda/direita
    R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    center = (c_le + c_re) / 2
    rot = lambda p: (p - center) @ R.T
    face, lb, rb, lt, rt, le, re, ul, ll, nb = map(rot, (face, lb, rb, lt, rt, le, re, ul, ll, nb))

    # 2) medidas-base
    y_top, y_chin = face[:, 1].min(), face[:, 1].max()
    H = y_chin - y_top
    W = face[:, 0].max() - face[:, 0].min()
    f = {}
    f['h_w'] = H / W

    # 3) posições verticais das referências, como fração da altura (0 = topo, 1 = queixo)
    brow_top = np.vstack([lt, rt])[:, 1].min()
    brow_y = np.vstack([lb, rb])[:, 1].mean()
    eye_y = np.vstack([le, re])[:, 1].mean()
    nose_y = nb[:, 1].mean()
    mouth_y = ul[:, 1].min()
    mouth_c = (ul[:, 1].min() + ll[:, 1].max()) / 2
    pos = lambda y: (y - y_top) / H
    f['pos_sobrancelha'] = pos(brow_y)
    f['pos_olho'] = pos(eye_y)
    f['pos_nariz'] = pos(nose_y)
    f['pos_boca'] = pos(mouth_c)
    # terços do rosto
    f['terco_testa'] = pos(brow_y)
    f['terco_meio'] = pos(nose_y) - pos(brow_y)
    f['terco_baixo'] = 1 - pos(nose_y)
    f['boca_queixo'] = (y_chin - ll[:, 1].max()) / H

    # 4) perfil de largura em cada altura (largura / W)
    #for p in np.arange(0.10, 0.96, 0.05):
    #    f[f'larg_{int(round(p*100)):02d}'] = width_at(face, y_top + p * H) / W


    # larguras nas referências anatômicas
    f['larg_testa_sobr'] = width_at(face, brow_y - 0.5*(brow_y - y_top)) / W   # meio da testa
    f['larg_sobrancelha'] = width_at(face, brow_y) / W
    f['larg_olho'] = width_at(face, eye_y) / W                                  # ~maçãs do rosto
    f['larg_nariz'] = width_at(face, nose_y) / W
    f['larg_boca'] = width_at(face, mouth_c) / W
    f['larg_mandibula'] = width_at(face, (mouth_c + y_chin) / 2) / W
    f['larg_queixo'] = width_at(face, y_chin - 0.08*H) / W

    # 5) razões entre larguras (testa vs maçã vs mandíbula)
    f['testa_maca'] = f['larg_sobrancelha'] / f['larg_olho']
    f['mandib_maca'] = f['larg_mandibula'] / f['larg_olho']
    f['mandib_testa'] = f['larg_mandibula'] / f['larg_sobrancelha']
    f['queixo_mandib'] = f['larg_queixo'] / f['larg_mandibula']
    f['maca_vs_max'] = f['larg_olho']  # já em relação a W

    # 6) ângulos do queixo / mandíbula (pontos 0..35, queixo ≈ 18)
    f['ang_queixo'] = angle(face[14], face[18], face[22])
    f['ang_queixo_largo'] = angle(face[12], face[18], face[24])
    f['ang_mand_dir'] = angle(face[9], face[13], face[18])
    f['ang_mand_esq'] = angle(face[27], face[23], face[18])
    f['ang_mand'] = (f['ang_mand_dir'] + f['ang_mand_esq']) / 2
    # 7) curvatura / "quadratice": área do contorno vs retângulo e vs elipse
    x, y = face[:, 0], face[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    f['area_ret'] = area / (W * H)                 # elipse≈0.785, retângulo=1
    per = np.sum(np.linalg.norm(face - np.roll(face, 1, axis=0), axis=1))
    f['circularidade'] = 4 * np.pi * area / per**2
    # distância olhos vs largura
    f['olhos_larg'] = np.linalg.norm(c_le - c_re) / W
    return f

def build(path):
    df = load(path)
    X = pd.DataFrame([extract(r) for _, r in df.iterrows()])
    return df, X