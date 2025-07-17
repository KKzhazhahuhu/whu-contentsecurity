import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 读取音频文件
audio_path = "babycrying.wav"
datal, sampling_ratel = librosa.load(audio_path)  # datal返回numpy数组，sr是采样频率
print(f"Audio loaded: {audio_path}")

# 绘制波形图
plt.figure(figsize=(10, 4))
librosa.display.waveshow(datal, sr=sampling_ratel)
plt.title('Waveform')
plt.show()

# 绘制标准声谱图
plt.figure(figsize=(12, 4))
D = librosa.stft(datal)  # 计算复数频谱
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # 幅度转分贝
librosa.display.specshow(S_db, sr=sampling_ratel,
                         x_axis='time', y_axis='linear',
                         hop_length=512, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Short-Time Fourier Transform Spectrogram')
plt.ylabel('frequency (Hz)')
plt.show()

# 绘制线性频率功率谱图
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(datal)), ref=np.max)
librosa.display.specshow(D, y_axis='linear', sr=sampling_ratel)
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

# 绘制Mel频谱图
plt.figure(figsize=(10, 4))
spectrogram = librosa.feature.melspectrogram(y=datal, sr=sampling_ratel)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
librosa.display.specshow(spectrogram_db, sr=sampling_ratel, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

# 绘制Mel倒谱图(MFCC)
plt.figure(figsize=(10, 4))
mel_ceps = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram))
librosa.display.specshow(mel_ceps, x_axis='time')
plt.colorbar()
plt.title('MFCC (Mel Cepstrum)')
plt.show()