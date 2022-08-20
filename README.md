# CPlus-pallet
Command line utility that generates a color pallet based on an image inputed by the user. The amount of colors can also be specified.

# Usage

The program has two inputs, a path to an image file (it can be a png or a jpg), and how many colors you want in your
pallet.

For example:

```bash
$ ./cpluspallet ~/Pictures/apicture.png 8
```
will generate 8 colors based off the colors in apicture.png.

You can have the program generate as many colors you want

# Building
You will need the CImg in order to build this project.

You can follow the install instructions from CImg's main page: 
https://cimg.eu/download.html

Or if you are on linux, you can install it using the package manager of your choice.

On arch, you can install CImg from the AUR with the following:
```bash
$ yay CImg
```
and choose the option *community/cimg*.

Once you have gotten CImg, you can now compile and link the project.

```bash 
$ g++ -o cpluspallet src/main.cpp src/colors.cpp -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
```
 Running this will create a binary called 'cpluspallet'.

# References

These stack overflow articles:
https://stackoverflow.com/questions/14375156/how-to-convert-a-rgb-color-value-to-an-hexadecimal-value-in-c
https://stackoverflow.com/questions/5823854/how-can-i-generate-a-palette-of-prominent-colors-from-an-image

And these:
https://sighack.com/post/averaging-rgb-colors-the-right-way
https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-cpp/
