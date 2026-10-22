# ------------------------------------------------------------------------
# [MSDet カスタム] マルチスペクトル TIFF 対応モジュール
# ------------------------------------------------------------------------
"""マルチスペクトル (多バンド) TIFF を RF-DETR で扱うためのヘルパー.

RF-DETR 本体は画像を常に ``PIL.Image.convert("RGB")`` で 3ch に変換するため,
G / R / NIR / RE のような 4ch 以上の TIFF を読むと 1 ページ目しか使われない.
このモジュールは TIFF を ``(C, H, W)`` テンソルとして読み込む処理をまとめたもの.

ライブラリ側の変更箇所はすべて ``# [MSDet TIFF]`` コメントで印を付けてある:

* ``datasets/yolo.py``        : TIFF をテンソルで読み込む / labels ディレクトリの探索を拡張
* ``datasets/transforms.py``  : ``Normalize`` の mean/std をチャンネル数に合わせて自動拡張
* ``training/module_data.py`` : GPU (Kornia) 正規化の mean/std をチャンネル数に合わせる
* ``detr.py``                 : ``predict("xxx.tif")`` で TIFF を直接推論できるようにする

対応する TIFF 形式:

1. マルチページ TIFF (1 ページ = 1 バンド).  ``cv2.imwritemulti`` の出力がこれ.
2. 1 ページに複数サンプルを持つ TIFF (``(H, W, C)``).

uint8 はそのまま uint8 で返す (後段の ``ToDtype(scale=True)`` で 0-1 に変換される).
uint16 などそれ以外は dtype の最大値で割って float32 (0-1) にして返す.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageSequence

TIFF_EXTENSIONS = {".tif", ".tiff"}


def is_tiff(path: str | Path) -> bool:
    """拡張子が ``.tif`` / ``.tiff`` かどうかを返す."""
    return Path(path).suffix.lower() in TIFF_EXTENSIONS


# ------------------------------------------------------------------------
# 読み込み
# ------------------------------------------------------------------------
def _read_with_cv2(path: Path) -> np.ndarray | None:
    """OpenCV で TIFF を ``(C, H, W)`` 配列として読む (imwritemulti と最も親和性が高い)."""
    try:
        import cv2

        ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if ok and len(frames) > 0:
            if len(frames) == 1:
                f = frames[0]
                if f.ndim == 2:
                    return f[np.newaxis]
                return np.transpose(f, (2, 0, 1))
            return np.stack(frames, axis=0)
    except Exception:
        pass
    return None


def _read_with_pillow(path: Path) -> np.ndarray:
    """Pillow で TIFF を ``(C, H, W)`` 配列として読む."""
    with Image.open(path) as image:
        n_frames = getattr(image, "n_frames", 1)
        if n_frames > 1:
            # マルチページ: 各ページを 1 バンドとして積み重ねる
            bands = [np.array(frame) for frame in ImageSequence.Iterator(image)]
            if any(band.ndim != 2 for band in bands):
                raise ValueError(f"マルチページ TIFF の各ページは 1 バンドである必要があります: {path}")
            return np.stack(bands, axis=0)
        # シングルページ: (H, W) または (H, W, C)
        array = np.array(image)
    if array.ndim == 2:
        return array[np.newaxis]
    return np.transpose(array, (2, 0, 1))


def _read_with_tifffile(path: Path) -> np.ndarray:
    """Pillow で読めない TIFF (5 バンド以上のシングルページ等) 用のフォールバック."""
    import tifffile  # 必要な時だけ import

    array = tifffile.imread(str(path))
    if array.ndim == 2:
        return array[np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"未対応の TIFF 形状です {array.shape}: {path}")
    # チャンネル軸は一番小さい次元とみなす ((C,H,W) と (H,W,C) の両方に対応)
    if array.shape[-1] < array.shape[0]:
        array = np.transpose(array, (2, 0, 1))
    return array


def read_tiff_bands(path: str | Path) -> np.ndarray:
    """TIFF を ``(C, H, W)`` の numpy 配列として読み込む.

    cv2.imreadmulti -> Pillow -> tifffile の順で試行し, 確実にデコードする.

    Args:
        path: TIFF ファイルのパス.

    Returns:
        ``(C, H, W)`` 配列 (dtype はファイルのまま).
    """
    path = Path(path)
    cv2_bands = _read_with_cv2(path)
    if cv2_bands is not None:
        return np.ascontiguousarray(cv2_bands)

    try:
        bands = _read_with_pillow(path)
    except (OSError, ValueError):
        bands = _read_with_tifffile(path)
    return np.ascontiguousarray(bands)



def load_tiff_as_tensor(path: str | Path, expected_channels: int | None = None) -> torch.Tensor:
    """TIFF を学習パイプライン用の ``(C, H, W)`` テンソルとして読み込む.

    Args:
        path: TIFF ファイルのパス.
        expected_channels: 指定した場合, チャンネル数が一致しなければエラーにする.

    Returns:
        uint8 の場合は uint8 テンソル, それ以外は 0-1 の float32 テンソル.

    Raises:
        ValueError: チャンネル数が ``expected_channels`` と一致しない場合.
    """
    bands = read_tiff_bands(path)

    if expected_channels is not None and bands.shape[0] != expected_channels:
        raise ValueError(
            f"TIFF のチャンネル数 ({bands.shape[0]}) がモデルの num_channels ({expected_channels}) と"
            f"一致しません: {path}\n"
            f"  -> RFDETRxxx(num_channels={bands.shape[0]}) のようにモデル作成時に指定してください."
        )

    if bands.dtype == np.uint8:
        return torch.from_numpy(bands)

    # uint16 / float など: 0-1 の float32 に変換
    if np.issubdtype(bands.dtype, np.integer):
        scale = float(np.iinfo(bands.dtype).max)
        return torch.from_numpy(bands.astype(np.float32) / scale)
    return torch.from_numpy(bands.astype(np.float32)).clamp_(0.0, 1.0)


# ------------------------------------------------------------------------
# 正規化
# ------------------------------------------------------------------------
def expand_to_channels(values: Sequence[float], num_channels: int) -> list[float]:
    """3ch 用の mean/std を ``num_channels`` 個に拡張する (R,G,B,R,G,B,... と繰り返す).

    ``detr.py`` の推論時の処理と同じ規則にすることで, 学習時と推論時の正規化を揃える.

    Examples:
        >>> expand_to_channels([0.485, 0.456, 0.406], 4)
        [0.485, 0.456, 0.406, 0.485]
    """
    return [value for _, value in zip(range(num_channels), cycle(values))]


# ------------------------------------------------------------------------
# モデル適応 (パッチ埋め込みのチャンネル数拡張)
# ------------------------------------------------------------------------
def adapt_patch_embeddings(nn_model: torch.nn.Module, num_channels: int) -> None:
    """モデルのバックボーンパッチ埋め込みを任意チャンネル数に適応させる."""
    if num_channels == 3:
        return
    import copy
    from rfdetr.inference import _adapt_input_conv

    backbone = nn_model.backbone[0]
    # DINOv2 の patch_embeddings projection を取得
    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        return
    inner_encoder = getattr(encoder, "encoder", None)
    if inner_encoder is None:
        inner_encoder = getattr(getattr(encoder, "base_model", None), "model", None)
        if inner_encoder is not None:
            inner_encoder = getattr(inner_encoder, "encoder", None)
    if inner_encoder is None:
        return

    embeddings = getattr(inner_encoder, "embeddings", None)
    if embeddings is None:
        return
    patch_embed = getattr(embeddings, "patch_embeddings", None)
    if patch_embed is None:
        return

    proj = patch_embed.projection
    if proj.in_channels == num_channels:
        return

    new_proj = copy.deepcopy(proj)
    new_proj.in_channels = num_channels
    new_weight = _adapt_input_conv(num_channels, proj.weight)
    new_proj.weight = torch.nn.Parameter(new_weight)
    new_proj.weight.requires_grad = proj.weight.requires_grad
    patch_embed.projection = new_proj
    patch_embed.num_channels = num_channels

