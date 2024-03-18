import numpy as np
import matplotlib.pyplot as plt

# Çizgileri tanımla
def h11(x1, x2):
    return 4 - 2 * x1 - x2

def h12(x1, x2):
    return 1 - x1 - x2

def h13(x1, x2):
    return 3 + 2 * x1 - x2

# Çizgileri çiz
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)

X1, X2 = np.meshgrid(x1, x2)

plt.plot(x1, h11(x1, 0), label='h11: 4 - 2(X1) - (X2) = 0')
plt.plot(x1, h12(x1, 0), label='h12: 1 - (X1) - (X2) = 0')
plt.plot(x1, h13(x1, 0), label='h13: 3 + 2(X1) - (X2) = 0')

# Bölgeleri görselleştir
plt.fill_between(x1, h12(x1, 0), h13(x1, 0), where=(h12(x1, 0) <= h13(x1, 0)), color='lightpink', alpha=0.5, label='C2 Sınıfı Bölgesi')
plt.fill_between(x1, h11(x1, 0), h12(x1, 0), where=(h11(x1, 0) <= h12(x1, 0)), color='lightblue', alpha=0.5, label='C1 Sınıfı Bölgesi')

# Eksenleri ve etiketleri ayarla
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('C1 ve C2 Sınıfları için Bölgeler')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend(loc='upper left')
plt.axis('equal')
plt.show()
