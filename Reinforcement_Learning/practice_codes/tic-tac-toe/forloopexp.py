import numpy as np

LENGTH = 3
env = np.array([[1, 0, 2],
                [0, 1, 0],
                [2, 0, 1]])

possible_moves = []
for i in range(LENGTH):
    for j in range(LENGTH):
        print(f"i = {i}, j = {j}")  # Her bir iterasyon noktasını yazdır
        if env[i, j] == 0:  # Eğer hücre boşsa (0 ise)
            print(f"Hücre ({i}, {j}) boş, possible_moves listesine ekleniyor.")
            possible_moves.append((i, j))
        else:
            print(f"Hücre ({i}, {j}) dolu, işleme devam ediliyor.")

if possible_moves:  # Eğer boş hücre varsa
    idx = np.random.choice(len(possible_moves))  # Rastgele bir boş hücre seç
    next_move = possible_moves[idx]
    print(f"Seçilen hareket: {next_move}")
else:
    print("Boş hücre yok")
