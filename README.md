# CPlus-palette
Command line tool that generates color palettes based on an image given by the user.


# Usage
```
 cpluspalette: pathToImage numberOfColors -t [quantization type]
Example: cpluspalette ~/Pictures/picture.png 8 -t 1

-t 1: uses K mean Clustering for Color Palette Generation: slower but produces better palettes most of the time
-t 2: used Median Cut for Color Palette Generation: Faster than K means Clustering but color palettes aren't always as good
```
To generate a palette, input a path to an image file, the number of colors you want, and optionally the quantization type:
-t 1: Uses K means Clustering for color quantization
-t 2: Uses Median Cut for color quantization

By default, Cpluspalette will use K means clustering unless you specify a different quantization type. Generally, the palette quality using K means Clustering is better than what Medain Cut produces but is slower. However, depending on the image, K means Clustering may create a poor palette; in that case, try again using Medain Cut.

## Examples

Here is an example:
```bash
 cpluspalette 1261697.jpg 8
```
or 
```bash
 cpluspalette 1261697.jpg 8 -t 1
```
Both ways of running cpluspalette will generate a color palette of 8 colors using K means Clustering: 
```
Generating a 8 color palette from 1261697.jpg... 
Using K Mean Clustering::: 
#f0e889 
#efe289 
#ebe386 
#ebe281 
#b5d09b 
#b4cc92 
#1a646f 
#164454
```
Here are the following colors from the list above:

To use Median Cut, run:
```bash
 cpluspalette 1261697.jpg 8 -t 2
```
Which will generate the following:
```
Using MedianCut::: 
#825b41 
#9e9066 
#d1d590 
#1c4852 
#c3dfba 
#5fa091 
#185763 
#0e2a39
```
Here are the colors from the list above:

# Building

### What you need
#### Packages
You will need [CImg](https://www.cimg.eu/index.html), [ImageMagick](https://imagemagick.org/index.php), and the X11 headers installed on your system to build this project.

#### Compilers and Build tools
You will also need Cmake and the g++ compiler.


The following section includes installation instructions for these packages for different Linux distros.


### Arch and Arch-based Distros

On Arch, you can install CImg from the AUR with the following:
```bash
yay CImg
```
and choose the option *community/cimg*.

Then install ImageMagick, which is available in the official arch repositories:
```bash

sudo pacman -S imagemagick
```
If you do not have the X11 headers, you can install them from the official repository:
```bash
sudo pacman -S libx11

```
### Debian and Debian-Based Distros

On Debian (or mint, ubuntu, or other Debian-based distros), install ImageMagick:
```bash
sudo apt install ImageMagick
```
 CImg:
```bash
sudo apt install cimg-dev
```
And the X11 headers, if you don't have them:
```bash
sudo apt install libx11-dev
```

### Fedora and Fedora-Based Distros

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

Once you have gotten CImg, you can compile and link the project.

First, create a directory called build in the GitHub repo and then cd into it:
```bash
mkdir build && cd build
```

Next, configure Cmake and then build the project:
```bash
cmake ../
cmake --build .
```

Then if you want to install it on your system rather than having the executable
in the build directory, run the following:
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
