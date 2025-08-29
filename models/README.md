# Pre-trained Models

Download the pre-trained models and place them in this directory.

## Required Models:

### 1. LIP Dataset Model (20 classes)

- **File:** `exp-schp-201908261155-lip.pth`
- **Classes:** Background, Hat, Hair, Glove, Sunglasses, Upper-clothes, Dress, Coat, Socks, Pants, Jumpsuits, Scarf, Skirt, Face, Left-arm, Right-arm, Left-leg, Right-leg, Left-shoe, Right-shoe
- **Use Case:** Detailed human parsing for research

### 2. ATR Dataset Model (18 classes)

- **File:** `exp-schp-201908301523-atr.pth`
- **Classes:** Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf
- **Use Case:** Fashion and clothing analysis

### 3. Pascal Person Part Model (7 classes)

- **File:** `exp-schp-201908270938-pascal-person-part.pth`
- **Classes:** Background, Head, Torso, Upper Arms, Lower Arms, Upper Legs, Lower Legs
- **Use Case:** Basic body part segmentation, real-time applications

## Download Instructions:

Please refer to the original repository for model download links:
https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

## File Structure:

```
models/
├── exp-schp-201908261155-lip.pth
├── exp-schp-201908301523-atr.pth
└── exp-schp-201908270938-pascal-person-part.pth
```
