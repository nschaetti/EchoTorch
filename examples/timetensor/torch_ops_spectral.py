
import torch
import echotorch


x = torch.randn(10, 1000)
out = torch.stft(x, n_fft=20, hop_length=5, win_length=10)
# print(out)
print(out.size())

y = echotorch.randn(length=1000)
out = torch.stft(y, n_fft=20, hop_length=5, win_length=10)
# print(out)
print(out.size())

y = echotorch.randn(length=1000, batch_size=(10,))
out = torch.stft(y, n_fft=20, hop_length=5, win_length=10)
# print(out)
print(out.size())

z = torch.istft(out, n_fft=20, hop_length=5, win_length=10)
print("z: {}".format(z))
print(z.size())

x = torch.randn(10, 1000)
out = torch.bartlett_window(window_length=10)
print("bartlett_window: {}".format(out))
