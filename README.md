# CPlus-palette
Command line utility that generates a color palette based on an image inputted by the user. The amount of colors can also be specified.

# Usage

The program has two inputs, a path to an image file (it can be a png or a jpg), and how many colors you want in your
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


It also works with other image files as well:
This was saved as a jpg image.
![7255636](https://user-images.githubusercontent.com/90001607/185774868-909d7f8c-19cc-4272-a304-63691b3d5d2b.jpg)
##### (I got this image off wallpaperaccess: https://wallpaperaccess.com/3440x1440-minimal, scroll down a bit and you'll find it.)

and here are 8 colors generated:

![#2f3245](https://user-images.githubusercontent.com/90001607/187530226-1d9da401-857b-4221-be74-453ccefb2624.png)
![#3b3c53](https://user-images.githubusercontent.com/90001607/187530230-dbfb7ebf-6bf5-46c2-83ef-fff7793d905c.png)
![#171b21](https://user-images.githubusercontent.com/90001607/187530231-b147ecf8-d875-4dd2-a04d-55470d471222.png)
![#15171c](https://user-images.githubusercontent.com/90001607/187530234-24f71430-9c01-47c0-8e49-44713de8fa61.png)
![#141517](https://user-images.githubusercontent.com/90001607/187530235-fd598e1e-931c-41a8-b39f-36c29c4a9b5f.png)
![#232634](https://user-images.githubusercontent.com/90001607/187530236-10b515f8-863a-45fd-b63e-896c03eae54d.png)
![#c1757b](https://user-images.githubusercontent.com/90001607/187530237-78f70ec5-0b5d-4da8-88fd-7db57c665e3f.png)
![#da8e8b](https://user-images.githubusercontent.com/90001607/187530239-888ca422-7365-46b8-b002-165f000ce818.png)



You can have the program generate as many colors you want

# Building

### What you need

You will need [CImg](https://www.cimg.eu/index.html), [ImageMagick](https://imagemagick.org/index.php), and the X11 headers installed on your system in order to build this project.



Installation instructions for these packages for different linux distros are included in this README. 
You will also need Cmake and a c++ compiler to build (i.e g++, clang, etc..).

## Arch and Arch based Distros

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
## Debian and Debian Based Distros

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

## Fedora and Fedora Based Distros

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

### now that you have everything

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
