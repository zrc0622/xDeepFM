import torch
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import History

# 直接引用 TensorFlow/Keras 的基类
EarlyStopping = EarlyStopping
History = History

class ModelCheckpoint(ModelCheckpoint):
    """
    每个 Epoch 结束后保存模型的 Callback。
    继承自 Keras 的 ModelCheckpoint，但修改了保存逻辑以适配 PyTorch。

    Args:
        filepath: 保存路径字符串，可以包含格式化参数（如 {epoch:02d}, {val_loss:.2f}）。
        monitor: 监控的指标名称（如 'val_auc', 'val_loss'）。
        save_best_only: 如果为 True，仅当监控指标改善时才覆盖保存。
        save_weights_only: 如果为 True，仅保存 state_dict()；否则保存整个模型对象。
        mode: 'auto', 'min', 'max'。决定指标是越小越好还是越大越好。
        period: 检查点保存的间隔 Epoch 数。
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        # 检查是否达到保存周期
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # 格式化文件名，填入 epoch 和 logs 中的指标值
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            
            if self.save_best_only:
                # 获取当前指标值
                current = logs.get(self.monitor)
                if current is None:
                    print('Can save best model only with %s available, skipping.' % self.monitor)
                else:
                    # self.monitor_op 是根据 mode ('min'/'max') 自动选择的比较函数
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        # 更新最佳值
                        self.best = current
                        
                        # === PyTorch 保存逻辑 ===
                        if self.save_weights_only:
                            # 推荐：仅保存权重字典
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            # 保存整个模型（包含结构），依赖 pickle
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                # save_best_only=False 时，每个周期都保存
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)