import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    img = mpimg.imread("menu.jpg")  # image read karo
    plt.imshow(img)  # show karne ke liye
    plt.axis('off')  # axis hide karo
    plt.show()

main()
