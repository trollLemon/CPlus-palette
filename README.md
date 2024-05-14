# CPlus-palette
Command line tool that generates color palettes based on an image given by the user.


# Usage
```
Usage:
 ./cpluspalette: pathToImage -d [numberOfColors] -t [quantization type] <FORMAT>[' ','RGB']
Examples: ./cpluspalette ~/Pictures/picture.png 8 -k RGB
          ./cpluspalette~/Pictures/picture.png 12 -m

quantization types:
-k: uses K mean Clustering for Color Palette Generation: slower but produces better palettes most of the time
-m : used Median Cut for Color Palette Generation: Faster than K mean Clustering but color palettes aren't always as good
FORMAT types:
' ' leave empty for hex color codes
"RGB" for additional RGB color values along with the hex colors

```
To generate a palette, input a path to an image file, the number of colors you want, and optionally the quantization type:
-k: Uses K means Clustering for color quantization
-m: Uses Median Cut for color quantization

You can also specify additional formats of the color palette. For example:
```bash
./cpluspalette  ~/Pictures/Wallpapers/1.jpg 8 -k RGB   
```
will output a color palette in hex color codes and rgb values.

By default, Cpluspalette will use K means clustering unless you specify a different quantization type. Generally, the palette quality using K means Clustering is better than what Medain Cut produces but is slower. However, depending on the image, K means Clustering may create a poor palette; in that case, try again using Medain Cut.

## Examples

Here is an example. The image in this example was gotten from https://wallpaperaccess.com/abstract-minimalist:

![1261697](https://user-images.githubusercontent.com/90001607/224535970-b3313613-cba6-4618-83dc-09cda2df71fe.jpg)


```bash
 cpluspalette 1261697.jpg 8
```
or 
```bash
 cpluspalette 1261697.jpg 8 -k
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

![1](https://user-images.githubusercontent.com/90001607/224536280-080897c1-5b90-4ce4-a7ce-3f53c6e98a49.png)
![2](https://user-images.githubusercontent.com/90001607/224536281-0d5bbca2-f567-4a80-9e51-0a228e9404db.png)
![3](https://user-images.githubusercontent.com/90001607/224536282-283cee5c-c4b6-4c28-97f3-5791979f85d1.png)
![4](https://user-images.githubusercontent.com/90001607/224536283-cbd72a7a-1e61-47df-9603-d5e0bb64011b.png)
![5](https://user-images.githubusercontent.com/90001607/224536284-45f682cf-08c2-48f5-9ac6-9b3cc9ff2e08.png)
![6](https://user-images.githubusercontent.com/90001607/224536286-f41c1e89-1837-4f1e-a862-eacc80a21c99.png)
![7](https://user-images.githubusercontent.com/90001607/224536288-48b0a2c3-1cb2-41b1-939f-e7c8b420471d.png)
![8](https://user-images.githubusercontent.com/90001607/224536289-ff4cda00-ea7f-4c00-a867-e7ad90e59be8.png)

To use Median Cut, run:
```bash
 cpluspalette 1261697.jpg 8 -m
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

![1](https://user-images.githubusercontent.com/90001607/224536475-a6ccecd0-7f75-42ca-be8b-b39f972a2147.png)
![2](https://user-images.githubusercontent.com/90001607/224536476-c5387073-ffe6-4a1f-a5fe-9d6ba37d69ab.png)
![3](https://user-images.githubusercontent.com/90001607/224536477-d3ce5b2e-d55d-4350-8146-cb6ba2bd0c23.png)
![4](https://user-images.githubusercontent.com/90001607/224536478-e9595b1c-30c6-40f6-9fde-e0d7835ceac5.png)
![5](https://user-images.githubusercontent.com/90001607/224536479-c7038024-7b4c-4a27-8c74-e806844169c4.png)
![6](https://user-images.githubusercontent.com/90001607/224536480-6712fbe0-8d3c-4229-a749-e919883d3e20.png)
![7](https://user-images.githubusercontent.com/90001607/224536482-32588be4-7ffa-46b5-bbab-ec10d68465c8.png)
![8](https://user-images.githubusercontent.com/90001607/224536483-105b2a1b-ea03-40ad-ad24-2de66d5f22ec.png)

Here are some more images with the generated palettes and color quantization type:
![alena-aenami-endless-1k](https://user-images.githubusercontent.com/90001607/224536600-bdf0c8a8-5832-43c9-9bdb-6c3eb9b52960.jpg)
(image from: https://www.artstation.com/artwork/4bX4eY )

K mean Clustering:

![1](https://user-images.githubusercontent.com/90001607/224536770-474d5fb8-b2b1-491a-bf44-e78c0e86e9ee.png)
![2](https://user-images.githubusercontent.com/90001607/224536772-78e5e4f7-a5e0-41fc-b58b-ebe021b9a6c3.png)
![3](https://user-images.githubusercontent.com/90001607/224536773-f5c16c19-d103-45c1-ae46-ed319859824c.png)
![4](https://user-images.githubusercontent.com/90001607/224536774-a915b921-8d6d-444f-8bb6-bdba195cb327.png)
![5](https://user-images.githubusercontent.com/90001607/224536775-96a3e5a3-fdde-4f7e-bf07-4663890f6295.png)
![6](https://user-images.githubusercontent.com/90001607/224536776-45a69ea6-84c1-42ee-8a8a-ba87d45e8474.png)
![7](https://user-images.githubusercontent.com/90001607/224536777-8ff2e756-f565-469e-96c8-7dba7f46a9c6.png)
![8](https://user-images.githubusercontent.com/90001607/224536778-9a9c8caa-85d7-4bfa-a169-36e2e4703bd7.png)

_______________________________________________________________________________________________________________

![2test](https://user-images.githubusercontent.com/90001607/224842789-ee3dd660-78ac-4ed3-9cda-979dbc3c6442.png)

(image from: https://www.reddit.com/r/WidescreenWallpaper/comments/qy5dvn/abstracts_5160x2160/)


Median Cut::

![1](https://user-images.githubusercontent.com/90001607/224843431-b0216f26-9e87-48a8-a179-c1fd267d8b3b.png)
![2](https://user-images.githubusercontent.com/90001607/224843434-ffbc4131-7258-48a9-9f24-39296f2fc546.png)
![3](https://user-images.githubusercontent.com/90001607/224843436-905053bc-9f0b-4832-9c1e-40e6f11da1a6.png)
![4](https://user-images.githubusercontent.com/90001607/224843437-a0e4cd99-434e-4325-9088-c89188745795.png)

________________________________________________________________________________________________________________

# Building

### What you need
#### Packages

All dependencies are included in this repo.

#### Compilers and Build tools
You will need Cmake and a C++ compiler.

### Building

#### Linux

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
#### Windows



# References
http://ijimt.org/papers/102-M480.pdf
https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
https://curiousily.com/posts/color-palette-extraction-with-k-means-clustering/
https://github.com/ndrake127/kMeans
https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
https://sighack.com/post/averaging-rgb-colors-the-right-way
https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-cpp/
https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
