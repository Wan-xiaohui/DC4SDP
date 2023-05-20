# DC4SDP
Supplementary code and data of our paper being submitted to TOSEM entitled *Data Complexity: A New Perspective for Analyzing the Difficulty of Defect Prediction Tasks*

Defect prediction is crucial for software quality assurance and has been extensively researched over recent decades. However, prior studies rarely focus on data complexity in defect prediction tasks, and even less on understanding the difficulties of these tasks from the perspective of data complexity. In this paper, we conduct an empirical study to estimate the hardness of over 33,000 instances, employing a set of measures to characterize the inherent difficulty of instances and the characteristics of defect datasets. Our findings indicate that: (1) instance hardness in both classes displays a right-skewed distribution, with the defective class exhibiting a more scattered distribution; (2) class overlap is the primary factor influencing instance hardness and can be characterized through feature, structural, instance, and multiresolution overlap; (3) no universal preprocessing technique is applicable to all datasets, and it may not consistently reduce data complexity, fortunately, dataset complexity measures can help identify suitable techniques for specific datasets; (4) integrating data complexity information into the learning process can enhance an algorithm's learning capacity. In summary, this empirical study highlights the crucial role of data complexity in defect prediction tasks, and provides a novel perspective for advancing research in defect prediction techniques.

# Acknowledgements
Part of the code references the code from the following code repositories. We are very grateful for the excellent work of the authors of these repositories:
https://github.com/w4k2/problexity
https://github.com/ai-se/early-defect-prediction-tse
https://gitlab.com/ita-ml/pyhard
