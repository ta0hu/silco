## Instruction

- tested under torch==1.0, python==3.5, gcc>=5.4, nvcc==9.0
- download pretrained weight from [Google Drive](https://drive.google.com/drive/folders/1_oXoaXa5trAMXmH3fs1gVc-sGBRSl5xW?usp=sharing), put them in the root path
- conda create -n cl  python=3.5.6 cudatoolkit=9.0
- python install cocoapi


# Run
```
python train_det_ssd_silco.py
```

## Difference between ICCV version

- change 2 dataset subset to 4 dataset subset for more stable performance comparison
- use faster evaluation method, while the logic keeps the same
- more stable evaluation(episode-based)
- pytorch upgrade from torch0.4 to torch1.0

