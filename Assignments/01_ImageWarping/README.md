
## Implementation of Image Geometric Transformation

This repository is Zhijun Wang's implementation of Assignment_01 of DIP. 

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```


## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results
### Basic Transformation
- scale
  | ![image-20240930185701939](pics/scale0.png) | ![scale1](pics/scale1.png) |
  | ------------------------------------------- | -------------------------- |
  | ![scale2](pics/scale2.png)                  | ![scale3](pics/scale3.png) |
  
- rotation
  | ![ro0](pics/rotation0.png) | ![ro1](pics/rotation1.png) |
  | -------------------------- | -------------------------- |
  | ![ro2](pics/rotation2.png) | ![ro3](pics/rotation3.png) |

  



- translation
  | ![trans0](pics/translation0.png) | ![trans1](pics/translation1.png) |
  | -------------------------------- | -------------------------------- |
  | ![trans2](pics/translation2.png) | ![trans3](pics/translation3.png) |

- flip
  | ![flip0](pics/translation2.png) | ![flip1](pics/flip.png) |
  | ------------------------------- | ----------------------- |

  



### Point Guided Deformation:

| ![warp0](pics/warp0.png) | ![warp1](pics/warp1.png) |
| ------------------------ | ------------------------ |



## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
