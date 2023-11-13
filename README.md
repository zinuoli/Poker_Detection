# Bridge Card Detection for Identifying Cheating

<b>This project aims to provide technical support for bridge card games by detecting the suit, rank, position, and orientation of playing cards. The project is based on the YOLOv7 framework and collects data from various scenarios. We have also developed methods to synthesize data to improve the model’s accuracy. In the end, we achieved an impressive mAP@0.5 score of 0.959, indicating that our model exhibits excellent accuracy.</b>

<div>
<span class="author-block">
  <a href='https://hoplee6.github.io/'>Haopeng Li</a><sup> 1 </sup>
</span>,
  <span class="author-block">
    <a href='https://zinuoli.github.io/'>Zinuo Li</a><sup> 2 </sup>
  </span> and
  <span class="author-block">
    <a href="https://research.monash.edu/en/persons/qiuhong-ke" target="_blank">Qiuhong Ke</a><sup> 2 </sup>
  </span>
</span>
</div>

<b>1 University of Melbourne, 2 Monash University</b>

The author sequence grouped by institutions.

[Report](https://drive.google.com/file/d/1cv6HpQf7eOi5dAcdiIFNgA4EwON8eJOj/view?usp=sharing) | [Dataset](https://1drv.ms/u/s!AglHNUXUeno-hDQ_soEb_aWLIjEm?e=0bTYDm) | [Demo Video](https://drive.google.com/file/d/1M-vq04_nFCIuJdH-3sbhTt_wfKZIyP7l/view?usp=sharing) | [Weights (Card)](https://drive.google.com/file/d/1WbRC7j9wM36FmfNsIzc1e2-Xtx-kP6Ho/view?usp=sharing) | [Weights (Orientation)](https://drive.google.com/file/d/1Q51nyhbVRoN9_pE614bpP4RYzZ3Jc0Hi/view?usp=sharing)
---
![image](https://github.com/zinuoli/Poker_Detection/assets/94612909/9545a00d-52ba-4c46-ac81-58285d02fca3)

The label consists of two parts: the number and suit of the card, followed by the angle. For example, 2H 81 means this card is 2 Heart, and the angle between this card and the table edge closest to it is 81 degrees. B means back. For more information, please see <a href='https://drive.google.com/file/d/1cv6HpQf7eOi5dAcdiIFNgA4EwON8eJOj/view?usp=sharing'>Report.</a>

# ⚙️ Usage
To run this script, the **environment** should be first installed. Please check requirements.txt and download <a href='https://drive.google.com/file/d/18pO7Vzpr9MN__jMfB9bR3V_WUoGNKvDP/view?usp=sharing'>multi-object-tracker.</a>
```
pip install -r requirements.txt
cd multi-object-tracker
pip install -e .
```
**Run script according to the following steps:**
1. In **card_detect** project specify **save_dir** in **video_split.py**, run it to gain the screenshots of video in a directory ⟨D⟩.
2. In **card_detect** project, run:
```
 python detect.py -–weights card_detection.pt –-source ⟨D⟩ –-name ⟨N⟩ –-save-txt -–nosave –-save-conf
```
3. Find corresponding txt results in **card_detect/runs/detect/⟨N⟩/labels.**
4. Copy ⟨D⟩ and ⟨N⟩/labels above to **orientation_detect** project, and rename /labels to ⟨D⟩_res. For example, ⟨D⟩ is test_video so the ⟨D⟩_res will be test_video_res.
5. In seg table infer project, run:
```
python -u segment/predict.py -–weights orientation_detection.pt -–source ⟨D⟩ –-name <specify_your_name_here>
```
6. In **orientation_detect** project, find image results in orientation_detect/runs/predict-seg/⟨N⟩/vis.
7. In **orientation_detect** project, specify image folder in merge **images.py** and run it to obtain final video result.

# 💗 Acknowledgements
This project used the code implementation of <a href="https://github.com/WongKinYiu/yolov7">YOLOv7</a>, we appreciate their great work. If you are looking for more applications, please refer to them.
