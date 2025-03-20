# MSPFusion: A feature transformer for multidimensional spectral-polarization image fusion


[[`Paper`](https://doi.org/10.1016/j.eswa.2025.127079)] [[`Dataset`](https://drive.google.com/file/d/1wU2D9SXr5OsATYwjgJkelbg_95pAsMoW/view?usp=drive_link)] 

MSPFusion design
![MSPFusion design](images/Model_Diagram.png)

The **MSPFusion model** can fuse multi-source images such as spectral polarization. It has been verified on a spectral polarization [dataset](https://drive.google.com/file/d/1wU2D9SXr5OsATYwjgJkelbg_95pAsMoW/view?usp=drive_link) NWPUSP containing 34 scenes.

Data preview
![Data_Preview](images/Data_Preview.png)

NWPUSPI data
![NWPUSPI data](images/NWPUSP.png)

## Installation


Install MSPFusion:

```
pip install git+https://github.com/tgg-77/MSPFusion.git
```

or clone the repository locally and install with


```
pip install -r requirements.txt 
```


## <a name="Train"></a>Train

```
python train.py
```

## <a name="Test"></a>Test

```
python test_MSPFusion.py
```

## <a name="Result"></a>Result

![Result](images/Result.png)

![Radar_Chart](images/Radar_Chart.png)

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Citing MSPFusion

If you use MSPFusion or NWPUSP dataset in your research, please use the following BibTeX entry. 

```
@article{tong2025mspfusion,
  title={MSPFusion: A feature transformer for multidimensional Spectral-Polarization image fusion},
  author={Tong, Geng and Yao, Xinling and Li, Ben and Fu, Jiaye and Wang, Yan and Hao, Jia and Karim, Shahid and Yu, Yiting},
  journal={Expert Systems with Applications},
  pages={127079},
  year={2025},
  publisher={Elsevier}
}
```
