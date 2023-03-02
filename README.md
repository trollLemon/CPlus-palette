# CPlus-palette
Command line utility that generates a color palette based on an image inputted by the user.

# Usage
The program has two inputs, a path to an image and how many colors you want in your palette.

For example:

```bash
cpluspalette ~/Pictures/example.png 5
```
will generate 5 colors based off the colors in example.png:

![example](https://user-images.githubusercontent.com/90001607/187527831-1b01609d-0846-4d59-afc9-a698982a06a0.png)


These are the colors generated from the program:

![1](https://user-images.githubusercontent.com/90001607/208357978-9f34397a-2721-4f77-bf75-e46b400bc8e8.png)
![2](https://user-images.githubusercontent.com/90001607/208357981-378d2543-0d80-478c-b313-30c7f5e40b98.png)
![3](https://user-images.githubusercontent.com/90001607/208357983-0119f5c6-b90c-4519-9d8e-166727a0fb11.png)
![4](https://user-images.githubusercontent.com/90001607/208357984-08bff1cf-7f94-44c5-bb18-4ab63691be1c.png)
![5](https://user-images.githubusercontent.com/90001607/208357985-16c56095-7193-4e96-936b-06b1a808bc34.png)




# Building

### What you need
#### Packages
You will need [CImg](https://www.cimg.eu/index.html), [ImageMagick](https://imagemagick.org/index.php), and the X11 headers installed on your system in order to build this project.

#### Compilers and Build tools
You will also need Cmake and a c++ compiler to build (i.e g++, clang, etc..).


Installation instructions for these packages for different linux distros are included in this README. 


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
http://ijimt.org/papers/102-M480.pdf
https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
https://curiousily.com/posts/color-palette-extraction-with-k-means-clustering/
https://github.com/ndrake127/kMeans
https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
https://sighack.com/post/averaging-rgb-colors-the-right-way
https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-cpp/
https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
