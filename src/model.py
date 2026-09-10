import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time
import copy
import ssl
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_auc_score, accuracy_score

# Fix SSL para macOS sin certificados instalados
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class EMA:
    """
    Clase para mantener la Media Móvil Exponencial (EMA) de los pesos del modelo.
    Ayuda a estabilizar y mejorar el rendimiento de la evaluación en entrenamiento.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Registrar los pesos iniciales de los parámetros entrenables
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model: nn.Module):
        """Actualiza los pesos sombra (shadow) usando la fórmula EMA."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self, model: nn.Module):
        """Aplica los pesos de EMA al modelo para validación/inferencia."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self, model: nn.Module):
        """Restaura los pesos originales del modelo tras la evaluación."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data = self.original[name]
        self.original = {}


class MelanomaClassifier(nn.Module):
    """
    Clasificador de melanoma basado en EfficientNet-B4.
    """
    def __init__(self, freeze_strategy: str = 'partial'):
        super(MelanomaClassifier, self).__init__()
        
        # Cargar EfficientNet-B4 con pesos pre-entrenados en ImageNet
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)
        
        # Guardamos la red base (extrae características)
        self.features = model.features
        
        # Dimensión de las características para B4 es 1792
        in_features = 1792
        
        # Cabeza de clasificación personalizada
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Salida única para BCEWithLogitsLoss
        )
        
        # Inicialización de Xavier para la cabeza del clasificador
        self._initialize_weights()
        
        # Aplicamos estrategia de congelación inicial
        self._apply_freeze_strategy(freeze_strategy)
        
        self._print_model_stats()
        
    def _initialize_weights(self):
        """Inicializa los pesos de la cabeza de clasificación."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def _apply_freeze_strategy(self, strategy: str):
        """Aplica la estrategia de congelación a los parámetros de las características."""
        if strategy == 'all':
            for param in self.features.parameters():
                param.requires_grad = False
        elif strategy == 'partial':
            # Congela los primeros 5 bloques (features[0:5])
            for block in self.features[:5]:
                for param in block.parameters():
                    param.requires_grad = False
            # Descongela del bloque 5 en adelante
            for block in self.features[5:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif strategy == 'none':
            for param in self.features.parameters():
                param.requires_grad = True

        self._freeze_all_batchnorm()

    def unfreeze_to_depth(self, depth: int):
        """
        Descongelación progresiva por profundidad (bloques desde el final).
        - depth=0: solo se entrena el clasificador.
        - depth=1: features[-1] + clasificador.
        - depth=2: features[-2:] + clasificador, etc.
        """
        # Primero congela todas las características
        for param in self.features.parameters():
            param.requires_grad = False

        if depth > 0:
            for block in self.features[-depth:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Las BatchNorm se mantienen SIEMPRE congeladas (afín + running stats),
        # incluso dentro de bloques recién descongelados. Con 1.8% de prevalencia
        # maligna, muchos mini-batches apenas contienen positivos: dejar que las
        # BN reestimen sus estadísticas o su escala/desplazamiento en ese régimen
        # introduce ruido que se suma al del propio shock de descongelar pesos.
        self._freeze_all_batchnorm()

        print(f"[*] Descongelación progresiva aplicada. Profundidad: {depth}")
        self._print_model_stats()

    def _freeze_all_batchnorm(self):
        """
        Congela TODAS las BatchNorm del backbone: las pone en eval() (para que
        running_mean/running_var no se actualicen) y fija requires_grad=False
        en su weight/bias (para que tampoco reciban gradiente). Se aplica sin
        importar si el bloque que las contiene está congelado o no.
        """
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False

    def train(self, mode: bool = True):
        """
        Override: nn.Module.train(True) reactiva el modo entrenamiento de TODOS
        los submódulos, incluidas las BatchNorm de bloques congelados. Tras
        llamar a la implementación base, volvemos a congelar esas BN para que
        sus estadísticas (running_mean/var) no se muevan mientras sus pesos
        siguen frozen.
        """
        super().train(mode)
        if mode:
            self._freeze_all_batchnorm()
        return self
        
    def get_features_layer(self):
        """Retorna la última capa convolucional, útil para Grad-CAM."""
        return self.features[-1]
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
        
    def _print_model_stats(self):
        """Imprime las estadísticas básicas de los parámetros del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")


def build_optimizer_and_scheduler(
    model: nn.Module, 
    lr_backbone: float = 1e-4, 
    lr_head: float = 1e-3, 
    weight_decay: float = 1e-4, 
    n_epochs: int = 60, 
    warmup_epochs: int = 5, 
    eta_min: float = 1e-7
) -> Tuple[optim.Optimizer, Any]:
    """
    Construye el optimizador AdamW con tasas de aprendizaje diferenciadas
    y un scheduler secuencial (Warmup Linear -> Cosine Annealing).
    """
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # CosineAnnealingLR: decaimiento monótono hasta eta_min a lo largo de las
    # épocas restantes. A diferencia de CosineAnnealingWarmRestarts, el LR
    # nunca vuelve a subir a un pico — evita que un "warm restart" coincida
    # con una descongelación progresiva y desestabilice el backbone dos veces
    # a la vez (LR alto + bloques recién liberados).
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(n_epochs - warmup_epochs, 1),
        eta_min=eta_min
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    return optimizer, scheduler


class ModelTrainer:
    """
    Clase encargada de entrenar y validar el modelo.
    Incluye acumulación de gradientes, EMA, descongelación progresiva,
    precisión mixta y early stopping.
    """
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        scheduler: Any,
        device: torch.device = torch.device('cpu'),
        checkpoint_dir: str = './checkpoints',
        gradient_accumulation_steps: int = 4,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        label_smoothing: float = 0.1,
        patience: int = 15
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.ema = EMA(model, ema_decay) if use_ema else None
        self.label_smoothing = label_smoothing
        self.patience = patience
        
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_auc = 0.0
        self.best_model_weights = None
        self.lr_ramps: list = []  # mini warm-ups activos para bloques recién descongelados
        # LR base "de referencia" del grupo backbone, tomado de 'initial_lr'
        # (fijado por los schedulers en su __init__ y no modificado después).
        # Independiente de dónde esté el coseno en un momento dado: sirve de
        # ancla fija para escalar el LR de los grupos que se añaden al
        # descongelar más profundidad.
        backbone_group = self.optimizer.param_groups[0]
        self.base_backbone_lr = backbone_group.get('initial_lr', backbone_group['lr'])
        self.history: Dict[str, list] = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [], 
            'val_auc': [], 'lr': []
        }
        
    def _train_one_epoch(self, dataloader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            # Adaptamos las etiquetas a formato float para BCEWithLogitsLoss
            labels = labels.to(self.device).float().view(-1, 1)
            
            # Soporte para precisión mixta (solo CUDA)
            use_amp = 'cuda' in str(self.device)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Escalar la pérdida en base a la acumulación de gradientes
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            # Paso de optimización (y clipping) tras acumular gradientes
            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Actualizar EMA después del step
                if self.use_ema:
                    self.ema.update(self.model)
            
            # Acumulamos el loss real 
            running_loss += loss.item() * self.gradient_accumulation_steps * inputs.size(0)
            
            # Predicciones (labels pueden ser continuos por CutMix/MixUp)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            # Redondear labels para accuracy (CutMix/MixUp produce labels continuos)
            rounded_labels = (labels > 0.5).float()
            
            all_preds.extend(preds.detach().cpu().numpy().ravel())
            all_labels.extend(rounded_labels.cpu().numpy().ravel())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
        
    def _validate(self, dataloader) -> Tuple[float, float, float]:
        if self.use_ema:
            self.ema.apply_shadow(self.model)
            
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)
                
                use_amp = 'cuda' in str(self.device)
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                rounded_labels = (labels > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().ravel())
                all_preds.extend(preds.cpu().numpy().ravel())
                all_labels.extend(rounded_labels.cpu().numpy().ravel())
                
        # Restaurar tras la evaluación si se usó EMA
        if self.use_ema:
            self.ema.restore(self.model)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calcular AUC
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            # Por si solo hay una clase en el batch/dataset
            epoch_auc = 0.0
            
        return epoch_loss, epoch_acc, epoch_auc
        
    def _register_unfrozen_group(self, epoch_idx: int, depth: int, ramp_epochs: int = 3,
                                  start_frac: float = 0.1, depth_lr_decay: float = 0.5) -> None:
        """
        Tras un unfreeze_to_depth, detecta los parámetros que han pasado a
        requires_grad=True y que todavía no están en ningún grupo del
        optimizer. Los añade como grupo nuevo con LR reducido y lo deja
        marcado para un mini warm-up de `ramp_epochs` épocas, evitando el
        cold-start de AdamW (momentos exp_avg/exp_avg_sq arrancando a cero
        de golpe).

        LR discriminativo por profundidad: el `target_lr` NO se toma del
        valor "vivo" del scheduler (podía coincidir con un pico y contaminar
        el warm-up), sino que se fija como el lr_backbone base ya escalado
        por `depth_lr_decay ** (depth - 1)`. depth=1 descongela el bloque
        final (más específico de la tarea) y usa el LR base completo; cada
        paso de profundidad adicional libera bloques más genéricos/cercanos
        a la entrada y recibe un LR progresivamente menor, para no
        sobrescribir features de ImageNet que ya son útiles.
        """
        tracked_ids = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                tracked_ids.add(id(p))

        new_params = [p for p in self.model.features.parameters()
                      if p.requires_grad and id(p) not in tracked_ids]

        if not new_params:
            return

        base_backbone_lr = self.base_backbone_lr
        target_lr = base_backbone_lr * (depth_lr_decay ** max(depth - 1, 0))
        self.optimizer.add_param_group({'params': new_params, 'lr': target_lr * start_frac})

        self.lr_ramps.append({
            'group_idx': len(self.optimizer.param_groups) - 1,
            'target_lr': target_lr,
            'start_frac': start_frac,
            'start_epoch': epoch_idx,
            'ramp_epochs': ramp_epochs,
        })
        print(f"[*] Nuevo grupo de LR registrado (idx={self.lr_ramps[-1]['group_idx']}, "
              f"depth={depth}) con warm-up de {ramp_epochs} épocas hacia lr~{target_lr:.2e}")

    def _update_lr_ramps(self, epoch_idx: int) -> None:
        """
        Actualiza el LR de los grupos en warm-up. El scheduler principal
        (SequentialLR/Cosine) solo conoce los grupos originales, así que
        estos grupos "extra" se pilotan a mano aquí. Durante el warm-up
        interpolan linealmente desde start_frac*target_lr hasta su propio
        target_lr fijo (ya NO el LR vivo del grupo backbone, para no heredar
        picos). Una vez terminado el warm-up, el grupo sigue la misma forma
        relativa que el coseno principal: se escala por la fracción que
        representa el LR vivo del backbone respecto a su propio máximo.
        """
        if not self.lr_ramps:
            return
        backbone_lr_now = self.optimizer.param_groups[0]['lr']
        backbone_progress = backbone_lr_now / self.base_backbone_lr if self.base_backbone_lr > 0 else 1.0
        for ramp in self.lr_ramps:
            elapsed = epoch_idx - ramp['start_epoch']
            if elapsed < ramp['ramp_epochs']:
                progress = (elapsed + 1) / ramp['ramp_epochs']
                floor = ramp['target_lr'] * ramp['start_frac']
                lr = floor + (ramp['target_lr'] - floor) * progress
            else:
                # Tras el warm-up, sigue la misma forma de decaimiento relativo
                # que el coseno principal, pero escalada a su propio target_lr.
                lr = ramp['target_lr'] * backbone_progress
            self.optimizer.param_groups[ramp['group_idx']]['lr'] = lr

    def train(self, train_loader, val_loader, num_epochs: int,
              max_unfreeze_depth: int = 0, unfreeze_start_epoch: int = 1,
              unfreeze_interval: int = 8) -> Dict[str, list]:
        epochs_no_improve = 0
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

        # Calendario de descongelación progresiva. Cambios respecto a la v3.1:
        #  - Profundidad máxima limitada a `max_unfreeze_depth` (por defecto 0
        #    para este experimento, antes 2, antes 4): con max_unfreeze_depth=0
        #    solo se descongela la profundidad 0 en unfreeze_start_epoch (la
        #    cabeza/clasificador, ~923K params trainable de 18.5M totales) y
        #    el backbone permanece congelado el resto del entrenamiento —
        #    no hay más eventos de descongelación. Es la prueba de "solo
        #    entrenar la capa 0" para ver si el backbone congelado (features
        #    ImageNet sin adaptar) ya es suficiente o si el problema está en
        #    otro sitio (datos, umbral, cabeza).
        #  - Intervalo más amplio (`unfreeze_interval` épocas en vez de 5)
        #    para dar tiempo a que pesos/optimizer se asienten tras cada
        #    descongelación antes de la siguiente (sin efecto real aquí, ya
        #    que solo hay un evento de descongelación).
        unfreeze_schedule = {
            unfreeze_start_epoch + i * unfreeze_interval: i
            for i in range(max_unfreeze_depth + 1)
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 15)

            if hasattr(self.model, 'unfreeze_to_depth'):
                if (epoch + 1) in unfreeze_schedule:
                    depth = unfreeze_schedule[epoch + 1]
                    self.model.unfreeze_to_depth(depth)
                    if depth > 0:
                        # Mini warm-up de LR para los parámetros recién liberados
                        self._register_unfrozen_group(epoch_idx=epoch, depth=depth)

            start_time = time.time()
            
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc, val_auc = self._validate(val_loader)
            
            # Actualizar scheduler (grupos originales) y los mini warm-ups (grupos extra)
            self.scheduler.step()
            self._update_lr_ramps(epoch_idx=epoch)
            
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
            print(f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
            
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                if self.use_ema:
                    self.best_ema_weights = copy.deepcopy(self.ema.shadow)
                epochs_no_improve = 0
                # Guardar checkpoint
                self._save_checkpoint(epoch + 1, val_auc)
                print(f"[*] AUC validation mejorado a {self.best_auc:.4f}. Modelo guardado.")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.patience:
                print(f"Early stopping activado. Sin mejoras durante {self.patience} epochs.")
                break
                
        print("Entrenamiento completado. Restaurando los mejores pesos...")
        self.model.load_state_dict(self.best_model_weights)
        if self.use_ema and hasattr(self, 'best_ema_weights'):
            self.ema.shadow = self.best_ema_weights
            self.ema.apply_shadow(self.model)
            
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_auc: float) -> None:
        """Guarda el checkpoint del mejor modelo."""
        import os
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc": float(val_auc),
            "history": self.history,
        }
        if self.use_ema:
            checkpoint["ema_shadow"] = self.ema.shadow
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, path)

    def load_best_model(self) -> None:
        """Carga los pesos del mejor checkpoint para inferencia."""
        import os
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.use_ema and "ema_shadow" in checkpoint:
                self.ema.shadow = checkpoint["ema_shadow"]
                self.ema.apply_shadow(self.model)
            print(f"✅ Mejor modelo cargado (época {checkpoint['epoch']}, "
                  f"AUC={checkpoint['val_auc']:.4f})")
        else:
            print("⚠️  No se encontró checkpoint. Usando modelo actual.")

if __name__ == '__main__':
    # Test simple de inicialización y flujo forward
    print("Iniciando prueba rápida de MelanomaClassifier (EfficientNet-B4)...")
    model = MelanomaClassifier(freeze_strategy='partial')
    
    # Simular un batch (2 imágenes RGB de 224x224)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Shape de salida (esperado [2, 1]): {out.shape}")
    
    print("\nPrueba de descongelación progresiva (depth=0, depth=2):")
    model.unfreeze_to_depth(0)
    model.unfreeze_to_depth(2)
    
    print("\n[OK] El archivo model.py ha sido ejecutado/probado con éxito.")
