# Palette-based image decomposition, harmonization, and color transfer

**JIANCHAO TAN**, George Mason University  
**JOSE ECHEVARRIA**, Adobe Research  
**YOTAM GINGOLD**, George Mason University

## Abstract

We present a palette-based framework for color composition for visual applications. Color composition is a critical aspect of visual applications in art, design, and visualization. The color wheel is often used to explain pleasing color combinations in geometric terms, and, in digital design, to provide a user interface to visualize and manipulate colors.

We abstract relationships between palette colors as a compact set of axes describing harmonic templates over perceptually uniform color wheels. Our framework provides a basis for a variety of color-aware image operations, such as color harmonization and color transfer, and can be applied to videos.

To enable our approach, we introduce an extremely scalable and efficient yet simple palette-based image decomposition algorithm. Our approach is based on the geometry of images in RGBXY-space. This new geometric approach is orders of magnitude more efficient than previous work and requires no numerical optimization. We demonstrate a real-time layer decomposition tool. After preprocessing, our algorithm can decompose 6 MP images into layers in 20 milliseconds.

We also conducted three large-scale, wide-ranging perceptual studies on the perception of harmonic colors and harmonization algorithms.

## 1. Introduction

Color composition is critical in visual applications in art, design, and visualization. Over the centuries, different theories about how colors interact with each other have been proposed. While it is arguable whether a universal and comprehensive color theory will ever exist, most previous proposals share in common the use of a color wheel (with hue parameterized by angle) to explain pleasing color combinations in geometric terms. In the digital world, the color wheel often serves as a user interface to visualize and manipulate colors.

We define our color relationships in the CIE LCh color space (the cylindrical projection of CIE Lab). Contrary to previous work using HSV color wheels, the LCh color space ensures that perceptual effects are accounted for with no additional processing. For example, a simple global rotation of hue in LCh-space (but not HSV-space) preserves the perceived lightness or gradients in color themes and images.

To represent color information, we adopt the powerful palette-oriented point of view and propose to work with color palettes of arbitrary numbers of swatches. Unlike hue histograms, color palettes or swatches can come from a larger variety of sources (extracted from images, directly from user input, or from generative algorithms) and capture the 3D nature of LCh in a compact way.

## 2. Related Work

### 2.1 Color Harmonization

Many existing works have applied different concepts from traditional color theory for artists to improve the color composition of digital images. In their seminal paper, Cohen-Or et al. [2006] use hue histograms and harmonic templates defined as sectors of hue-saturation in HSV color space, to model and manipulate color relationships.

Instead of using hue histograms from images, our framework is built on top of color palettes, independently of their source. Given the higher level of abstraction of palettes, we simplify harmonic templates to arrangements of axes in chroma-hue space (from LCh), interpreted and derived directly from classical color theory. This more general and simpler representation makes for more intuitive metrics, easier to solve, that enable a wider range of applications.

### 2.2 Palette Extraction and Image Decomposition

**Palette Extraction**: A straightforward approach consists of using a k-means method to cluster the existing colors in an image, in RGB space. A different approach consists of computing and simplifying the convex hull enclosing all the color samples, which provides more general palettes that better represent the existing color gamut of the image.

**Image Decomposition**: For recoloring applications, it is also critical to find a mapping between the extracted color palette and the image pixels. Recent work is able to decompose the input image into separate layers according to a palette. We present a new, efficient method for layer decomposition, based on the additive color mixing model.

## 3. Palette Extraction and Image Decomposition

### 3.1 Palette Extraction

We extend Tan et al. [2016]'s work in two ways. First, we propose a simple, geometric layer decomposition method that is orders of magnitude more efficient than the state-of-the-art. Second, we propose a simple scheme for automatic palette size selection.

The convex hull of all pixel colors is computed and then simplified to a user-chosen palette size. We improve upon this procedure with the observation that the reconstruction error can be measured geometrically, even before layer decomposition, as the RMSE of every pixel's distance to the simplified convex hull.

We propose a simple automatic palette size selection based on a user-provided RMSE reconstruction error tolerance (2/255 in our experiments). For efficiency, we divide RGB-space into 32 × 32 × 32 bins. We measure the distance from each non-empty bin to the simplified convex hull, weighted by the bin count.

### 3.2 Image decomposition via RGBXY convex hull

In this work, we adopt linear mixing layers. We provide a fast and simple, yet spatially coherent, geometric construction.

Any point p inside a simplex has a unique set of barycentric coordinates, or convex additive mixing weights such that p = Σᵢ wᵢcᵢ, where the mixing weights wᵢ are positive and sum to one, and cᵢ are the vertices of the simplex.

#### Spatial Coherence

To provide spatial coherence, our key insight is to extend this approach to 5D RGBXY-space, where XY are the coordinates of a pixel in image space, so that spatial relationships are considered along with color in a unified way.

We first compute the convex hull of the image in RGBXY-space. We then compute Delaunay generalized barycentric coordinates (weights) for every pixel in the image in terms of the 5D convex hull. Pixels that have similar colors or are spatially adjacent will end up with similar weights, meaning that our layers will be smooth both in RGB and XY-space.

These mixing weights form a Q×N matrix W_RGBXY, where N is the number of image pixels and Q is the number of RGBXY convex hull vertices. We also compute W_RGB, Delaunay barycentric coordinates (weights) for the RGBXY convex hull vertices in the 3D simplified convex hull.

The final weights for the image are obtained via matrix multiplication: W = W_RGB W_RGBXY, which is a P × N matrix that assigns each pixel weights solely in terms of the simplified RGB convex hull.

#### Python Implementation

```python
from numpy import *
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix

def RGBXY_weights( RGB_palette, RGBXY_data ):
    RGBXY_hull_vertices = RGBXY_data[ ConvexHull( RGBXY_data ).vertices ]
    W_RGBXY = Delaunay_coordinates( RGBXY_hull_vertices, RGBXY_data )
    # Optional: Project outside RGBXY_hull_vertices[:,:3] onto RGB_palette convex hull.
    W_RGB = Star_coordinates( RGB_palette, RGBXY_hull_vertices[:,:3] )
    return W_RGBXY.dot( W_RGB )

def Star_coordinates( vertices, data ):
    ## Find the star vertex
    star = argmin( linalg.norm( vertices, axis=1 ) )
    ## Make a mesh for the palette
    hull = ConvexHull( vertices )
    ## Star tessellate the faces of the convex hull
    simplices = [ [star] + list(face) for face in hull.simplices if star not in face ]
    barycoords = -1*ones( ( data.shape[0], len(vertices) ) )
    ## Barycentric coordinates for the data in each simplex
    for s in simplices:
        s0 = vertices[s[:1]]
        b = linalg.solve( (vertices[s[1:]]-s0).T, (data-s0).T ).T
        b = append( 1-b.sum(axis=1)[:,None], b, axis=1 )
        ## Update barycoords whenever the data is inside the current simplex.
        mask = (b>=0).all(axis=1)
        barycoords[mask] = 0.
        barycoords[ix_(mask,s)] = b[mask]
    return barycoords

def Delaunay_coordinates( vertices, data ): # Adapted from Gareth Rees
    # Compute Delaunay tessellation.
    tri = Delaunay( vertices )
    # Find the tetrahedron containing each target (or -1 if not found).
    simplices = tri.find_simplex(data, tol=1e-6)
    assert (simplices != -1).all() # data contains outside vertices.
    # Affine transformation for simplex containing each datum.
    X = tri.transform[simplices, :data.shape[1]]
    # Offset of each datum from the origin of its simplex.
    Y = data - tri.transform[simplices, data.shape[1]]
    # Compute the barycentric coordinates of each datum in its simplex.
    b = einsum( '...jk,...k->...j', X, Y )
    barycoords = c_[b,1-b.sum(axis=1)]
    # Return the weights as a sparse matrix.
    rows = repeat(arange(len(data)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel()
    cols = tri.simplices[simplices].ravel()
    vals = barycoords.ravel()
    return coo_matrix( (vals,(rows,cols)), shape=(len(data),len(vertices)) ).tocsr()
```

#### Tessellation

To make the line of greys 2-sparse, the tessellation should ensure that an edge is created between the darkest and lightest color in the palette. We propose to use a star tessellation. If either a black or white palette color is chosen as the star vertex, it will form an edge with the other. We choose the darkest color in the palette as the star vertex.

### 3.3 Evaluation

**Quality**: The primary means to assess the quality of layers is to apply them for some purpose, such as recoloring, and then identify artifacts, such as noise, discontinuities, or surprisingly affected regions.

**Speed**: Our proposed RGBXY approach is orders of magnitude faster than previous methods. For 100 megapixel images, our approach took on average 12.6 minutes to compute with peak memory usage of 15 GB.

## 4. Color Harmonization

We describe our palette-based approach to color harmonization and color composition. We explain how we fit and enforce classical harmonic templates, and describe how our framework can be used for other color composition operations.

### 4.1 Template fitting

We use seven templates Tₘ, m = 1...7. A template is defined by Tⱼₘ(α), where j is the index of each axis and α is an angle of rotation in hue. We apply them in LCh-space (Lightness, Chroma, and hue) to match human perception.

Given an image I and its extracted color palette P, we seek to find the Tₘ(α) that is closest to the colors in P in the Ch plane. We define the distance D between a palette P and a template Tₘ(α) as:

D(P,Tₘ(α)) = Σᵢ₌₁|P| W(Pᵢ) · L(Pᵢ) · C(Pᵢ) · |H(Pᵢ) - Tⱼ*ₘ(α)|

where j* is the axis of template Tₘ(α) that is closest to palette color Pᵢ.

Since the search space is a finite range in 1D, we use a brute-force search to find the optimal global rotation angle α*ₘ:

α*ₘ = arg minₐ D(P,Tₘ(α))

Once a template is fit, we harmonize the input image by using Tₘ(α*ₘ) to move the colors in P closer to the axis assignment that minimizes equation 1. Users can control the strength of harmonization via an interpolation parameter, where β = 0 leaves the palette unchanged and β = 1 fully rotates each palette color to lie on its matched axis.

### 4.2 Beyond hue

Our compact representation using palettes and axis-based templates allows to formulate full 3D color harmonization operations easily.

**LC harmonization**: Apart from hue, some authors have described harmony in terms of lightness and chroma as well. While histogram-based approaches may be non-trivial to extend to these additional dimensions, our approach generalizes to them easily.

**Color-based contrast**: As part of his seminal work on color composition for design, Itten described additional pleasing color arrangements to create contrast. It is straightforward to model them with our axis-based representation.

## 5. Perceptual Study

We conducted a set of wide-ranging perceptual studies on harmonic colors and our harmonization algorithm. N = 616 participants took part in our studies with 31% self-reporting as having some knowledge in color theory.

All of our experiments are based on 2-alternative forced-choice (2AFC) questions. Participants were shown two images and asked to choose which of two images has the most harmonic colors.

### 5.1 Image and Palette Harmonization

The most notable observation about this first study is that participants overall preferred the original images to harmonizations and a preference for β = 0.5 to β = 1. Participants with knowledge about color theory had a statistically significant (p ≪ 0.001) stronger preference for harmonized images (3.7% overall).

### 5.2 Perception of Archetypal Color Harmony

The monochromatic and square templates were perceived to be significantly more harmonic than random palettes. However, random templates were perceived as more harmonic than complementary and triad templates.

## 6. Video Harmonization

Our methods can naturally extend to video by simply applying our image decomposition and harmonization on each frame independently. We first compute a global palette for each sequence of frames, aiming at a more coherent layer decomposition.

## 7. Color Transfer

Our palette extraction, image decomposition, and harmonic templates enable new approaches to color transfer.

**Template alignment**: Given an input image I and a reference image R, we extract their palettes P^I and P^R, and estimate their optimal templates, T_I(α*_I) and T_R(α*_R). We can compute the weight of each axis of the template and find the global rotation γ that aligns T_I with T_R.

**Template transfer**: When the final results should preserve better the original colors, we harmonize the input image colors P^I directly to the best-fitting template for the reference image T_R(α*_R), without any global rotation.

## 8. Conclusion

We have presented a very efficient, intuitive and capable framework for color composition. It allows us to formulate previous and novel approaches to color harmonization and color transfer with very robust results.

### 8.1 Limitations

During our performance tests for the image decomposition, we found isolated cases where the computation of the 5D convex hull takes somewhat longer than usual. We believe it is due to very specific color distributions, but we would like to study the phenomenon in more depth.

There are also cases for the templated color transfer where the input palette tries to match a reference palette with a higher number of axes. This is usually a case of colorization that we currently handle with varying success depending on the input color palette.

### 8.2 Future work

**Image decomposition**: We wish to explore the use of superpixels to see if we are able to achieve greater speed ups. We also wish to explore parallel and approximate convex hull algorithms.

**Other color-aware applications**: We believe that our templates may carry semantic structure that we would like to keep exploring in the future. This can enable higher level and more intuitive image search algorithms, where images or palettes can be used transparently to retrieve other images and color themes for design.
