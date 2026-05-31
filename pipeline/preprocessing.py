import cv2
import numpy as np

_VALID_METHODS = {
    "none",
    "gaussian",
    "clahe",
    "sharpen",
    "sobel_fusion",
    "canny_fusion",
}


def preprocess_image(img_rgb, method="none"):
    """
    Aplica pre-processamento classico opcional na imagem RGB antes da deteccao de vias.

    Parametros:
        img_rgb: imagem RGB em numpy array.
        method: nome do metodo de pre-processamento.

    Metodos disponiveis:
        - "none": retorna a imagem original.
        - "gaussian": suavizacao com filtro Gaussiano.
        - "clahe": equalizacao adaptativa de histograma no canal de luminosidade.
        - "sharpen": realce de nitidez usando unsharp mask.
        - "sobel_fusion": calcula Sobel e mistura suavemente com a imagem original.
        - "canny_fusion": calcula Canny e mistura suavemente com a imagem original.

    Retorna:
        imagem RGB pre-processada.
    """
    if method is None:
        method = "none"

    method = str(method).lower()
    if method not in _VALID_METHODS:
        options = ", ".join(sorted(_VALID_METHODS))
        raise ValueError(
            f"Metodo de pre-processamento invalido: {method}. Opcoes: {options}"
        )

    if method == "none":
        return img_rgb

    if method == "gaussian":
        return cv2.GaussianBlur(img_rgb, (3, 3), 0)

    if method == "clahe":
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge((l_chan, a_chan, b_chan))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if method == "sharpen":
        blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        return cv2.addWeighted(img_rgb, 1.5, blur, -0.5, 0)

    if method == "sobel_fusion":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        sobel_rgb = cv2.cvtColor(mag_norm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(img_rgb, 0.85, sobel_rgb, 0.15, 0)

    if method == "canny_fusion":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(img_rgb, 0.85, edges_rgb, 0.15, 0)

    return img_rgb
