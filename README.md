# CPlus-palette
Command line utility that generates a color palette based on an image inputted by the user. The amount of colors can also be specified.

# Usage

The program has two inputs, a path to an image file, and how many colors you want in your
palette.

For example:

```bash
cpluspalette ~/Pictures/example.png 5
```
will generate 5 colors based off the colors in example.png:

![example](https://user-images.githubusercontent.com/90001607/187527831-1b01609d-0846-4d59-afc9-a698982a06a0.png)

The output will look like the following:

![image](https://user-images.githubusercontent.com/90001607/187527765-840ba92d-d2e2-4c79-a548-3d9413be511a.png)

These are the colors generated from the program:


![#2a2e3a](https://user-images.githubusercontent.com/90001607/187529250-57aae882-766e-4ce7-a01e-776bc4f5aa42.png)
![#6e7597](https://user-images.githubusercontent.com/90001607/187529252-d95fd989-ba84-4c71-aeba-7975d8616e6a.png)
![#8e7a99](https://user-images.githubusercontent.com/90001607/187529254-42f6424b-68f0-4642-a47e-e362d836de75.png)
![#495371](https://user-images.githubusercontent.com/90001607/187529256-f59605ba-ccbc-4a01-a08e-ee46e2c2f6fd.png)
![#d1697b](https://user-images.githubusercontent.com/90001607/187529258-284d44ab-b04b-402f-98b7-6609c241a45d.png)



# Building

### What you need

You will need [CImg](https://www.cimg.eu/index.html), [ImageMagick](https://imagemagick.org/index.php), and the X11 headers installed on your system in order to build this project.



Installation instructions for these packages for different linux distros are included in this README. 
You will also need Cmake and a c++ compiler to build (i.e g++, clang, etc..).

### Arch and Arch based Distros

On arch, you can install CImg from the AUR with the following:
```bash
yay CImg
```
and choose the option *community/cimg*.

Then install imagemagick, it is available in the official arch repositories:
```bash

sudo pacman -S imagemagick
```
If you dont have the X11 headers you can install them from the official repository:
```bash
sudo pacman -S libx11

```
### Debian and Debian Based Distros

On Debian (or mint, ubuntu, other debian based distros), install imagemagick:
```bash
sudo apt install imagemagick
```
 CImg:
```bash
sudo apt install cimg-dev
```
and the X11 headers if you don't have them:
```bash
sudo apt install libx11-dev
```

### Fedora and Fedora Based Distros

X11 headers:
```bash
sudo yum install libX11-dev
```

Cimg:

```bash
sudo dnf install CImg-devel

```
ImageMagick:
```bash
sudo dnf install ImageMagick
```

## Compiling

Once you have gotten CImg, you can now compile and link the project.

First, create a directory called build in the Github repo and then cd into it:
```bash
mkdir build && cd build
```

next configure Cmake and then build the project:
```bash
cmake ../
cmake --build .
```

Then if you want to install it on your system rather than having the executable
in the build directory, run:
```bash
sudo make install
```


# References
https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/

https://curiousily.com/posts/color-palette-extraction-with-k-means-clustering/

https://github.com/ndrake127/kMeans

https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/

https://sighack.com/post/averaging-rgb-colors-the-right-way

https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-cpp/

https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
