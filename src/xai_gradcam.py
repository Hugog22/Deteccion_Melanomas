"""
================================================================================
FASE 3: INTELIGENCIA ARTIFICIAL EXPLICABLE (XAI) Y DETECCIÓN DEL SESGO
================================================================================
Módulo: xai_gradcam.py

Descripción:
    Implementa Grad-CAM y Grad-CAM++ desde cero, sin dependencias externas,
    para la interpretación de predicciones del modelo y la detección del efecto
    Clever Hans en imágenes dermatoscópicas.

Fundamento matemático de Grad-CAM (Selvaraju et al., 2017):
    Para una clase objetivo c y la última capa convolucional L:

    1. Calcular gradientes de la puntuación de clase y^c respecto a los
       feature maps A^k de la capa L:
       ∂y^c / ∂A^k_ij

    2. Ponderar por importancia (Global Average Pooling de gradientes):
       α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)

    3. Combinación lineal ponderada con ReLU:
       L^c_GradCAM = ReLU(Σ_k α^c_k · A^k)

Fundamento matemático de Grad-CAM++ (Chattopadhay et al., 2018):
    Mejora los pesos α^c_k usando derivadas de segundo orden para localizar
    mejor múltiples instancias de la misma clase en una imagen:

    1. Calcular α ponderado por segunda derivada del score:
       α^c_k_ij = (∂²y^c / ∂(A^k_ij)²) /
                  (2·∂²y^c/∂(A^k_ij)² + Σ A^k_ab · ∂³y^c/∂(A^k_ij)³)

       En la práctica (implementación eficiente):
       grad_sq = (∂y^c/∂A^k)²
       α = grad_sq / (2·grad_sq + Σ_spatial(A^k · grad_cube))

    2. Pesos finales:
       w^c_k = Σ_i Σ_j α^c_k_ij · ReLU(∂y^c / ∂A^k_ij)

    3. Mapa de calor: L^c_GradCAM++ = ReLU(Σ_k w^c_k · A^k)

    Para dermoscopía: Grad-CAM++ localiza mejor cuando hay múltiples regiones
    sospechosas en la imagen (artefactos periféricos + lesión central).

Detección del efecto Clever Hans:
    El sesgo Clever Hans ocurre cuando el modelo genera altas activaciones en
    regiones de la imagen que contienen artefactos (reglas, marcadores) en lugar
    de la lesión dermatoscópica. Grad-CAM++ permite detectarlo visualmente y
    cuantificarlo mediante la métrica de "Artefact Overlap Score" (AOS).

Referencias:
    Selvaraju RR et al. "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization." IEEE ICCV 2017. arXiv:1610.02391

    Chattopadhay A et al. "Grad-CAM++: Generalized Gradient-Based Visual
    Explanations for Deep Convolutional Networks." IEEE WACV 2018.
    arXiv:1710.11063

    Winkler JK et al. "Association Between Surgical Skin Markings in Dermoscopy
    Images and Diagnostic Performance." JAMA Dermatol. 2019;155(10):1135-1141.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3.1: IMPLEMENTACIÓN DE GRAD-CAM DESDE CERO
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Implementación de Grad-CAM (Selvaraju et al., 2017) compatible con
    cualquier arquitectura CNN de PyTorch.
    
    Funcionamiento mediante hooks de PyTorch:
        - Forward hook: Captura los feature maps A^k de la capa objetivo.
        - Backward hook: Captura los gradientes ∂y^c/∂A^k durante backprop.
    
    Ventajas sobre CAM original (Zhou et al., 2016):
        - No requiere Global Average Pooling en la arquitectura
        - Funciona con cualquier arquitectura CNN (ResNet, DenseNet, VGG...)
        - Los gradientes proveen importancia relativa de canales específicos
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        """
        Args:
            model: Modelo PyTorch evaluado.
            target_layer: Capa convolucional de la que extraer activaciones.
                         Para DenseNet-121: model.features.denseblock4
                         Para ResNet-50: model.model.layer4
        """
        self.model = model
        self.target_layer = target_layer
        
        # Contenedores para activaciones y gradientes (capturados por hooks)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Registro de hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """
        Registra hooks de PyTorch para capturar activaciones y gradientes.
        
        Forward hook: Almacena output de la capa (feature maps A^k)
        Backward hook: Almacena gradientes dL/dA^k tras backward()
        """
        def forward_hook(module, input, output):
            # Detach para evitar que los hooks formen parte del grafo
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: gradientes fluyendo hacia los feature maps
            self.gradients = grad_output[0].detach()
        
        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self) -> None:
        """Elimina los hooks para liberar memoria. Llamar después de Grad-CAM."""
        self._forward_handle.remove()
        self._backward_handle.remove()
    
    def _suppress_padding_artifacts(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Elimina artefactos de padding en los bordes. PyTorch EfficientNet
        tiene un bug conocido donde las convoluciones asimétricas acumulan
        activaciones inmensas en el borde derecho/inferior de los feature maps.
        """
        # cam shape: (1, 1, H', W')
        _, _, h, w = cam.shape
        if h > 2 and w > 2:
            cam[:, :, 0, :] = 0  # Top
            cam[:, :, -1, :] = 0 # Bottom
            cam[:, :, :, 0] = 0  # Left
            cam[:, :, :, -1] = 0 # Right
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Genera el mapa de activación Grad-CAM para la clase objetivo.
        
        Args:
            input_tensor: Tensor de imagen normalizada, shape (1, 3, H, W).
            target_class: Clase objetivo. Si None, usa la clase predicha.
        
        Returns:
            Tupla (heatmap, predicted_class, confidence):
                - heatmap: Array numpy (H, W) normalizado [0, 1]
                - predicted_class: Índice de clase predicha
                - confidence: Probabilidad de la clase objetivo [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        # ── Paso 1: Forward pass ──────────────────────────────────────────
        # El hook forward_hook captura automáticamente las activaciones A^k
        logits = self.model(input_tensor)
        
        # Compatibilidad: salida única (sigmoid) o 2 clases (softmax)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            # Modelo binario de salida única
            logit_val = logits.view(-1)[0]
            confidence = torch.sigmoid(logit_val).item()
            predicted_class = 1 if confidence > 0.5 else 0
            if target_class is None:
                target_class = predicted_class
            # Para backward, usamos el logit directamente (clase 1) o su negativo (clase 0)
            class_score = logit_val if target_class == 1 else -logit_val
        else:
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            if target_class is None:
                target_class = predicted_class
            confidence = probs[0, target_class].item()
            class_score = logits[0, target_class]
        
        # ── Paso 2: Backward pass para la clase objetivo ──────────────────
        self.model.zero_grad()
        class_score.backward(retain_graph=False)
        
        # Validar que los hooks capturaron datos
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Hooks no capturaron datos. Verificar que target_layer "
                "sea correcto y que el forward pass lo atraviese."
            )
        
        # ── Paso 3: Calcular α^c_k (pesos de importancia por canal) ──────
        # α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        # Global Average Pooling sobre dimensiones espaciales (H', W')
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # ── Paso 4: Combinación lineal ponderada ──────────────────────────
        # L^c_GradCAM = ReLU(Σ_k α^c_k · A^k)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Solo activaciones positivas (relevantes para la clase)
        cam = self._suppress_padding_artifacts(cam)

        
        # ── Paso 5: Interpolación al tamaño de la imagen original ─────────
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )
        
        # ── Paso 6: Normalización a [0, 1] ───────────────────────────────
        cam = cam.squeeze().cpu().numpy()
        cam = self._normalize_cam(cam)
        
        return cam, predicted_class, confidence
    
    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """
        Normaliza usando percentiles en lugar de min/max estricto.
        Evita que un solo píxel anómalo (outlier de padding) aplaste
        visualmente las activaciones legítimas de la lesión.
        """
        p_low, p_high = np.percentile(cam, [2, 98])
        if p_high - p_low > 1e-8:
            cam = np.clip((cam - p_low) / (p_high - p_low), 0, 1)
        else:
            cam = np.zeros_like(cam)
        return cam

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Superpone el mapa de calor Grad-CAM sobre la imagen original.
        
        Args:
            image: Imagen RGB original (H, W, 3), valores [0, 255].
            heatmap: Mapa de activación normalizado (H, W), valores [0, 1].
            alpha: Transparencia del heatmap (0=transparente, 1=opaco).
            colormap: Mapa de color OpenCV (JET es estándar en literatura XAI).
        
        Returns:
            Imagen con heatmap superpuesto (H, W, 3), valores [0, 255].
        """
        # Convertir heatmap a imagen de color (escala de grises → colormap)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blending: overlay = α * heatmap + (1-α) * imagen
        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        return overlay


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3.2: GRAD-CAM++ (MEJORA SOBRE GRAD-CAM)
# ─────────────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus(GradCAM):
    """
    Implementación de Grad-CAM++ (Chattopadhay et al., 2018).
    
    Mejora sobre Grad-CAM:
        Utiliza la segunda derivada (Hessian) del score de clase para
        ponderar los gradientes de forma más precisa, especialmente cuando
        hay múltiples instancias de una clase en la imagen.
    
    Para dermoscopía: Mejora la localización cuando hay múltiples regiones
    sospechosas en la misma imagen (múltiples focos de activación).
    
    Referencia:
        Chattopadhay A et al. "Grad-CAM++: Generalized Gradient-Based Visual
        Explanations for Deep Convolutional Networks." IEEE WACV 2018.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        
        # Compatibilidad: salida única (sigmoid) o 2 clases (softmax)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            logit_val = logits.view(-1)[0]
            confidence = torch.sigmoid(logit_val).item()
            predicted_class = 1 if confidence > 0.5 else 0
            if target_class is None:
                target_class = predicted_class
            class_score = logit_val if target_class == 1 else -logit_val
        else:
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            if target_class is None:
                target_class = predicted_class
            confidence = probs[0, target_class].item()
            class_score = logits[0, target_class]
        
        self.model.zero_grad()
        class_score.backward(retain_graph=False)
        
        # ── Grad-CAM++: Ponderación por segunda derivada ──────────────────
        gradients = self.gradients      # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # α = gradients^2 / (2 * gradients^2 + sum(activations * gradients^3))
        grad_sq = gradients ** 2
        grad_cube = gradients ** 3
        
        # Denominador: evitar división por cero
        denominator = 2 * grad_sq + (activations * grad_cube).sum(
            dim=(2, 3), keepdim=True
        )
        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator)
        )
        
        alpha = grad_sq / denominator
        
        # Pesos finales: α ponderado por ReLU(gradients)
        weights = (alpha * F.relu(gradients)).mean(dim=(2, 3), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self._suppress_padding_artifacts(cam)
        
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(input_h, input_w), mode="bilinear",
                           align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam = self._normalize_cam(cam)
        
        return cam, predicted_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3.3: DEMOSTRACIÓN DEL EFECTO CLEVER HANS
# ─────────────────────────────────────────────────────────────────────────────
class CleverHansAnalyzer:
    """
    Herramienta de análisis para detectar y visualizar el efecto Clever Hans
    en modelos de clasificación de lesiones dermatoscópicas.
    
    El efecto Clever Hans en dermoscopía:
        Winkler et al. (2019) demostraron que los modelos de clasificación de
        melanoma aprenden a asociar reglas de medición con diagnóstico maligno
        porque en el ISIC dataset histórico, las imágenes de melanomas enviadas
        por dermatólogos frecuentemente incluían estos artefactos como marcadores
        para biopsia. Esta correlación espuria es aprendida por el modelo.
    
    Esta clase permite:
        1. Visualizar mapas Grad-CAM sobre imágenes con artefactos
        2. Calcular el Artifact Overlap Score (AOS) como métrica cuantitativa
        3. Comparar modelo base vs. modelo mitigado
    """
    
    ARTIFACT_COLORS = {
        "ruler":      (0, 0, 0),      # Negro: regla de medición
        "ink_marker": (128, 0, 128),  # Morado: marcador quirúrgico
        "patch":      (200, 200, 180) # Beige: parche de gel
    }
    
    CLASS_NAMES = {0: "Benigno", 1: "Maligno"}
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
        use_gradcam_pp: bool = True
    ):
        """
        Args:
            model: Modelo a analizar.
            target_layer: Capa objetivo para Grad-CAM.
            device: Dispositivo de cómputo.
            use_gradcam_pp: Si True, usa Grad-CAM++; si False, Grad-CAM estándar.
        """
        self.model = model
        self.device = device
        CAMClass = GradCAMPlusPlus if use_gradcam_pp else GradCAM
        self.gradcam = CAMClass(model, target_layer)
        self.cam_name = "Grad-CAM++" if use_gradcam_pp else "Grad-CAM"
    
    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Revierte la normalización ImageNet para visualización.
        
        Operación inversa: x_orig = x_norm * std + mean
        """
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
        
        img = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img
    
    def compute_artifact_overlap_score(
        self,
        heatmap: np.ndarray,
        artifact_mask: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Calcula el Artifact Overlap Score (AOS): proporción de la energía
        de activación Grad-CAM que recae sobre regiones de artefactos.
        
        AOS = Σ_{p ∈ artefacto} CAM(p) / Σ_p CAM(p)
        
        AOS → 1: Modelo enfocado principalmente en artefactos (sesgo Clever Hans)
        AOS → 0: Modelo enfocado en la lesión (comportamiento deseado)
        
        Args:
            heatmap: Mapa Grad-CAM normalizado (H, W) ∈ [0, 1].
            artifact_mask: Máscara binaria (H, W) donde 1 = región de artefacto.
            threshold: Umbral para binarizar el heatmap.
        
        Returns:
            Artifact Overlap Score ∈ [0, 1].
        """
        total_activation = heatmap.sum()
        if total_activation < 1e-8:
            return 0.0
        
        artifact_activation = (heatmap * artifact_mask).sum()
        return float(artifact_activation / total_activation)
    def _generate_debiased_gradcam(self, inp: torch.Tensor, target_class: int = 1) -> Tuple[np.ndarray, int, float]:
        """
        Genera un Grad-CAM promediado sobre aumentos (TTA) para cancelar artefactos
        de borde fijos a las coordenadas del tensor.
        """
        transforms = [
            (lambda x: x, lambda c: c),
            (lambda x: torch.flip(x, dims=[3]), lambda c: np.fliplr(c)),
            (lambda x: torch.flip(x, dims=[2]), lambda c: np.flipud(c)),
            (lambda x: torch.flip(x, dims=[2, 3]), lambda c: np.flip(c, axis=(0, 1))),
        ]
        
        cams = []
        final_pred, final_conf = None, None
        
        for fwd, inv in transforms:
            cam, pred, conf = self.gradcam(fwd(inp.clone()), target_class)
            cams.append(inv(cam))
            if final_pred is None:
                final_pred, final_conf = pred, conf
                
        # Promedio y renormalización
        mean_cam = np.mean(cams, axis=0)
        p_low, p_high = np.percentile(mean_cam, [2, 98])
        if p_high - p_low > 1e-8:
            mean_cam = np.clip((mean_cam - p_low) / (p_high - p_low), 0, 1)
        else:
            mean_cam = np.zeros_like(mean_cam)
            
        return mean_cam, final_pred, final_conf

    def visualize_clever_hans_effect(
        self,
        images_with_artifacts: List[Tuple[torch.Tensor, int, str, str]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 5)
    ) -> plt.Figure:
        """
        Genera figura comparativa que muestra el efecto Clever Hans:
        Columnas: [Imagen Original | Mapa Grad-CAM | Overlay | Diagnóstico]
        
        Args:
            images_with_artifacts: Lista de tuplas
                (image_tensor, true_label, artifact_type, image_path).
            save_path: Ruta para guardar la figura.
            figsize: Tamaño de la figura matplotlib.
        
        Returns:
            Figura matplotlib con las visualizaciones.
        """
        n_images = len(images_with_artifacts)
        if n_images == 0:
            raise ValueError("Se requieren imágenes para visualizar el efecto Clever Hans.")
        fig = plt.figure(figsize=(figsize[0], figsize[1] * n_images))
        fig.patch.set_facecolor('#0a0a0a')
        
        outer = gridspec.GridSpec(n_images, 1, figure=fig, hspace=0.4)
        
        for img_idx, (img_tensor, true_label, artifact_type, img_path) in \
                enumerate(images_with_artifacts):
            
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=outer[img_idx], wspace=0.05
            )
            
            # Preparar tensor para Grad-CAM
            inp = img_tensor.unsqueeze(0).to(self.device)
            
            # Generar Grad-CAM con debias por TTA
            try:
                heatmap, pred_class, confidence = self._generate_debiased_gradcam(inp, target_class=1)
            except Exception as e:
                print(f"⚠️  Error en Grad-CAM para imagen {img_idx}: {e}")
                continue
            
            # Denormalizar imagen para visualización
            img_orig = self._denormalize_image(img_tensor)
            
            # Overlay del heatmap
            overlay = GradCAM.overlay_heatmap(img_orig, heatmap, alpha=0.45)
            
            # ── Panel 1: Imagen Original ──────────────────────────────────
            ax1 = fig.add_subplot(inner[0])
            ax1.imshow(img_orig)
            ax1.set_title(f"Original\nVerdad: {self.CLASS_NAMES[true_label]}",
                         color='white', fontsize=9, fontweight='bold')
            artifact_label = f"Artefacto: {artifact_type}" if artifact_type != "none" else "Sin artefacto"
            ax1.set_xlabel(artifact_label, color='#ff9f00', fontsize=8)
            ax1.set_xticks([])
            ax1.set_yticks([])
            for spine in ax1.spines.values():
                spine.set_edgecolor('#ff9f00' if artifact_type != "none" else '#444')
                spine.set_linewidth(2)
            
            # ── Panel 2: Mapa de Calor Grad-CAM ──────────────────────────
            ax2 = fig.add_subplot(inner[1])
            im = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
            ax2.set_title(f"{self.cam_name}\nActivación clase: Maligno",
                         color='white', fontsize=9)
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # Detectar si hay sesgo: zona de alta activación en región periférica
            h, w = heatmap.shape
            center_mask = np.zeros_like(heatmap)
            margin = int(min(h, w) * 0.15)
            center_mask[margin:-margin, margin:-margin] = 1
            peripheral_activation = heatmap[~center_mask.astype(bool)].mean()
            center_activation = heatmap[center_mask.astype(bool)].mean()
            bias_detected = peripheral_activation > center_activation * 0.8
            
            # ── Panel 3: Overlay Grad-CAM ─────────────────────────────────
            ax3 = fig.add_subplot(inner[2])
            ax3.imshow(overlay)
            status = "⚠️ SESGO DETECTADO" if bias_detected else "✓ Sin sesgo"
            status_color = '#ff4444' if bias_detected else '#44ff88'
            ax3.set_title(f"Overlay {self.cam_name}\n{status}",
                         color=status_color, fontsize=9, fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
            for spine in ax3.spines.values():
                spine.set_edgecolor(status_color)
                spine.set_linewidth(2.5)
            
            # ── Panel 4: Diagnóstico del Modelo ───────────────────────────
            ax4 = fig.add_subplot(inner[3])
            ax4.set_facecolor('#111')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_xticks([])
            ax4.set_yticks([])
            
            pred_name = self.CLASS_NAMES[pred_class]
            is_correct = pred_class == true_label
            result_color = '#44ff88' if is_correct else '#ff4444'
            result_text = "CORRECTO" if is_correct else "ERROR"
            pred_confidence = confidence if pred_class == 1 else (1.0 - confidence)
            
            ax4.text(0.5, 0.75, f"Pred: {pred_name}",
                    ha='center', va='center', color='white',
                    fontsize=11, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.5, 0.58, f"Conf. pred.: {pred_confidence:.1%}",
                    ha='center', va='center', color='#aaa',
                    fontsize=10, transform=ax4.transAxes)
            ax4.text(0.5, 0.46, f"P(maligno): {confidence:.1%}",
                    ha='center', va='center', color='#aaa',
                    fontsize=9, transform=ax4.transAxes)
            ax4.text(0.5, 0.35, result_text,
                    ha='center', va='center', color=result_color,
                    fontsize=12, fontweight='bold', transform=ax4.transAxes)
            
            # Score de activación periférica
            ax4.text(0.5, 0.15,
                    f"Act. periférica: {peripheral_activation:.3f}\n"
                    f"Act. central: {center_activation:.3f}",
                    ha='center', va='center', color='#888',
                    fontsize=8, transform=ax4.transAxes)
            ax4.set_title("Diagnóstico", color='white', fontsize=9)
        
        plt.suptitle(
            f"Análisis XAI — Detección de Sesgo Clever Hans\n"
            f"Técnica: {self.cam_name} | Modelo: EfficientNet-B4",
            color='white', fontsize=13, fontweight='bold', y=1.01
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150,
                       facecolor='#0a0a0a')
            print(f"✅ Figura guardada en: {save_path}")
        
        return fig
    
    @staticmethod
    def detect_artifacts_classical(image_path: str) -> Tuple[bool, str]:
        """
        Detecta artefactos clínicos en imágenes dermatoscópicas usando visión
        por computador clásica — completamente INDEPENDIENTE del modelo.

        Esta función es la fuente de la etiqueta has_artifact cuando los datos
        reales ISIC no tienen anotaciones de artefacto (DEMO_MODE=False).
        Al ser independiente del modelo, garantiza que la comparación de AOS
        entre grupos (con/sin artefacto) sea estadísticamente válida.

        Detecta dos tipos de artefacto prevalentes en ISIC histórico
        (Winkler et al. 2019):

        1. REGLAS DE MEDICIÓN (ruler): Líneas rectas oscuras paralelas
           típicamente en los bordes de la imagen.
           Detector: cv2.HoughLinesP con umbral de longitud mínima >= 20% del ancho.

        2. MARCADOR QUIRÚRGICO (ink_marker): Parches de color violeta/morado
           en el rango HSV [130, 50, 50] – [170, 255, 255].
           Detector: umbralización de color HSV (> 0.5% de píxeles afectados).

        Args:
            image_path: Ruta a la imagen dermatoscópica.

        Returns:
            Tupla (has_artifact: bool, artifact_type: str).
            artifact_type puede ser "ruler", "ink_marker" o "none".
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "none"
            h, w = img.shape[:2]

            # ── 1. Detector de reglas (líneas rectas oscuras) ─────────────────
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Realzar bordes
            edges = cv2.Canny(gray, threshold1=30, threshold2=80)
            # Detectar líneas largas (≥ 20% del ancho o alto)
            min_line_len = int(min(h, w) * 0.20)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=40,
                minLineLength=min_line_len,
                maxLineGap=8
            )
            if lines is not None and len(lines) >= 2:
                # Verificar que al menos 2 líneas sean aproximadamente paralelas
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
                angles = np.array(angles)
                # Agrupar ángulos similares (diferencia < 15°)
                for ang in angles:
                    parallel = np.sum(np.abs(angles - ang) < 15)
                    if parallel >= 2:
                        return True, "ruler"

            # ── 2. Detector de marcador quirúrgico (color violeta/morado) ─────
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Rango HSV para violeta/morado (marcador Gentian Violet típico en ISIC)
            lower_violet = np.array([125, 40, 40])
            upper_violet = np.array([165, 255, 255])
            mask_violet = cv2.inRange(hsv, lower_violet, upper_violet)
            violet_ratio = mask_violet.sum() / (255.0 * h * w)
            if violet_ratio > 0.005:  # más del 0.5% de píxeles
                return True, "ink_marker"

            return False, "none"

        except Exception:
            return False, "none"

    def batch_bias_analysis(
        self,
        loader: torch.utils.data.DataLoader,
        df_metadata: "pd.DataFrame",
        max_samples: int = 50
    ) -> Dict:
        """
        Análisis cuantitativo del sesgo Clever Hans en todo el conjunto de test.

        Compara métricas Grad-CAM (activación periférica / AOS) entre imágenes
        CON y SIN artefactos clínicos, usando la siguiente jerarquía de etiquetas:

          1. Etiqueta de metadatos (has_artifact=True): cuando hay inyección
             de artefactos sintéticos (DEMO_MODE=True). Etiqueta más fiable.
          2. Detector CV clásico independiente: cuando has_artifact=False para
             todos (datos reales, DEMO_MODE=False). Usa HoughLinesP + HSV color
             para detección de reglas y marcadores, completamente independiente
             del modelo.

        La independencia de la etiqueta respecto a la salida de Grad-CAM es
        CRÍTICA para que el Mann-Whitney U tenga validez estadística.

        Args:
            loader: DataLoader con return_path=True.
            df_metadata: DataFrame con metadatos (columnas has_artifact, image_path).
            max_samples: Número máximo de muestras a analizar.

        Returns:
            Diccionario con estadísticas descriptivas y listas '_aos_with' /
            '_aos_without' para StatisticalValidator.mann_whitney_bias().
            Incluye '_detection_mode': "metadata_labels" | "classical_cv".
        """
        self.model.eval()

        # ── Paso 1: Recopilar activaciones y etiquetas ────────────────────────
        all_records = []
        n_analyzed = 0

        for batch in loader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [""] * len(images)

            for i in range(len(images)):
                if n_analyzed >= max_samples:
                    break

                img_tensor = images[i:i+1].to(self.device)
                try:
                    heatmap, pred_class, confidence = self.gradcam(
                        img_tensor, target_class=1
                    )
                except Exception:
                    continue

                path_str = str(paths[i])
                has_artifact = False
                mask_path_str = None

                if path_str and df_metadata is not None:
                    matching = df_metadata[df_metadata["image_path"] == path_str]
                    if not matching.empty:
                        has_artifact = bool(matching.iloc[0]["has_artifact"])
                        if "artifact_mask_path" in matching.columns:
                            mask_path_str = matching.iloc[0].get("artifact_mask_path")

                # Métrica homogénea para comparación entre grupos:
                # activación periférica normalizada. Si existe máscara del artefacto,
                # además calculamos AOS solo como descriptor adicional.
                h, w = heatmap.shape
                total_act = max(heatmap.sum(), 1e-8)
                margin = int(min(h, w) * 0.15)
                center_mask = np.zeros_like(heatmap, dtype=bool)
                center_mask[margin:-margin, margin:-margin] = True
                peripheral_act = float(heatmap[~center_mask].sum() / total_act)
                center_act = float(heatmap[center_mask].sum() / total_act)
                artifact_overlap = None
                if has_artifact and mask_path_str and pd.notna(mask_path_str) and os.path.exists(str(mask_path_str)):
                    mask_img = Image.open(str(mask_path_str)).convert("L")
                    mask_img = mask_img.resize((w, h), Image.NEAREST)
                    mask_arr = np.array(mask_img) > 128
                    artifact_overlap = self.compute_artifact_overlap_score(heatmap, mask_arr)

                all_records.append({
                    "image_path": path_str,
                    "peripheral_act": peripheral_act,
                    "center_act": center_act,
                    "artifact_overlap": artifact_overlap,
                    "confidence": confidence,
                    "has_artifact": has_artifact,
                })
                n_analyzed += 1

        # ── Paso 2: Si no hay etiquetas de metadatos, aplicar detector CV ─────
        n_labeled = sum(r["has_artifact"] for r in all_records)
        detection_mode = "metadata_labels"

        if n_labeled == 0 and all_records:
            detection_mode = "classical_cv"
            print("\n  INFO: has_artifact=False para todos (DEMO_MODE=False, datos reales).")
            print("  INFO: Aplicando detector CV clasico independiente (HoughLines + HSV color)...")
            n_detected = 0
            for r in all_records:
                if r["image_path"]:
                    detected, atype = self.detect_artifacts_classical(r["image_path"])
                    r["has_artifact"] = detected
                    if detected:
                        n_detected += 1
            n_labeled_cv = sum(r["has_artifact"] for r in all_records)
            print(f"  CV detector: {n_labeled_cv}/{len(all_records)} imagenes con artefacto detectado")
            if n_labeled_cv == 0:
                print("  AVISO: El detector CV no encontro artefactos. El dataset ISIC descargado")
                print("  via API publica puede no contener imagenes con reglas/marcadores.")
                print("  Para un test valido, usa DEMO_MODE=True (ver docstring de ISICArchiveDownloader).")

        # ── Paso 3: Agrupar resultados ────────────────────────────────────────
        results = {
            "with_artifact": {
                "peripheral_act": [], "center_act": [], "confidence": [], "artifact_overlap": []
            },
            "without_artifact": {
                "peripheral_act": [], "center_act": [], "confidence": [], "artifact_overlap": []
            },
        }
        for r in all_records:
            key = "with_artifact" if r["has_artifact"] else "without_artifact"
            results[key]["peripheral_act"].append(r["peripheral_act"])
            results[key]["center_act"].append(r["center_act"])
            results[key]["confidence"].append(r["confidence"])
            if r["artifact_overlap"] is not None:
                results[key]["artifact_overlap"].append(r["artifact_overlap"])

        summary = {}
        for key, data in results.items():
            if data["peripheral_act"]:
                n = len(data["peripheral_act"])
                pa = np.array(data["peripheral_act"])
                ca = np.array(data["center_act"])
                summary[key] = {
                    "n_samples":             n,
                    "mean_peripheral_act":   float(pa.mean()),
                    "std_peripheral_act":    float(pa.std(ddof=1)) if n > 1 else 0.0,
                    "median_peripheral_act": float(np.median(pa)),
                    "mean_center_act":       float(ca.mean()),
                    "mean_confidence":       float(np.mean(data["confidence"])),
                    "bias_ratio":            float(pa.mean() / max(ca.mean(), 1e-8)),
                }
                if data["artifact_overlap"]:
                    ao = np.array(data["artifact_overlap"])
                    summary[key]["mean_artifact_overlap"] = float(ao.mean())
                    summary[key]["median_artifact_overlap"] = float(np.median(ao))

        summary["_mw_with"] = list(results["with_artifact"]["peripheral_act"])
        summary["_mw_without"] = list(results["without_artifact"]["peripheral_act"])
        summary["_aos_with"] = list(results["with_artifact"]["artifact_overlap"])
        summary["_detection_mode"] = detection_mode
        summary["_score_kind"] = "peripheral_act"

        # Diagnóstico
        print("\n📊 Analisis Descriptivo del Sesgo Clever Hans:")
        print(f"   (Modo deteccion: {detection_mode})")
        print("─" * 60)
        for key, stats in summary.items():
            if key.startswith("_"):
                continue
            label = "Con artefacto" if key == "with_artifact" else "Sin artefacto"
            print(f"\n  {label} (n={stats['n_samples']}):")
            print(f"    Act. periferica: {stats['mean_peripheral_act']:.4f} "
                  f"+/- {stats['std_peripheral_act']:.4f} "
                  f"(mediana {stats['median_peripheral_act']:.4f})")
            print(f"    Act. central:    {stats['mean_center_act']:.4f}")
            print(f"    Bias ratio:      {stats['bias_ratio']:.3f}")
            print(f"    Confianza media: {stats['mean_confidence']:.3f}")
            if "mean_artifact_overlap" in stats:
                print(f"    AOS medio:       {stats['mean_artifact_overlap']:.4f}")

        n_with = len(summary["_mw_with"])
        n_without = len(summary["_mw_without"])
        print(f"\n  Grupos para Mann-Whitney U: n_with={n_with}, n_without={n_without}")
        return summary
