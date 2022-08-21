# CPlus-pallete
Command line utility that generates a color pallet based on an image inputed by the user. The amount of colors can also be specified.

# Usage

The program has two inputs, a path to an image file (it can be a png or a jpg), and how many colors you want in your
pallete.

For example:

```bash
$ ./cpluspallet ~/Pictures/apicture.png 5
```
will generate 5 colors based off the colors in apicture.png.

The output will look like the following:

![image](https://user-images.githubusercontent.com/90001607/185774728-5e7a760c-e054-4ff5-837a-a7470e927a80.png)

These are colors generated from this image:

![image](https://user-images.githubusercontent.com/90001607/185774623-82d15335-8d70-444a-83ff-15c2b0006ec6.png)

These are the colors outputed from the program:

![#4aee30](https://user-images.githubusercontent.com/90001607/185774744-4b3ebd2d-4411-400c-ab88-cef6e5c59231.png)
![#7b80a1](https://user-images.githubusercontent.com/90001607/185774745-f2c34367-3c6d-46d7-a66b-e873e90791fe.png)
![#28ccff](https://user-images.githubusercontent.com/90001607/185774747-6cfe540f-4cad-4dfe-ae98-b1f7d618ee44.png)
![#44e21e](https://user-images.githubusercontent.com/90001607/185774748-837e0a14-7b68-484d-a085-c98438a6880c.png)
![#cab0db](https://user-images.githubusercontent.com/90001607/185774749-fef2b515-6b9b-4e64-b2ab-3607db21704d.png)


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

On Debain (or mint, ubuntu, other debian based distros), you will need to get imagemagick:
```bash
$  sudo apt install imagemagick
```
and CImg:
```bash
$  sudo apt install cimg-dev
```
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
