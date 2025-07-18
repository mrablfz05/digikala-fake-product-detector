<!-- Untill now we have a few problems that I recognized them in plots that should solved -->

<!-- 1.Rate_cnt: have a lots of 0 values that makes also Rate be 0. about 75% of this cols are 0. (5.output.png, 6.output.png and nonlogicalline.png)-->

<!-- 2.Imbalance class: our aproach is detecting fake products but with current col calls Is_Fake the model just can train on detecting which model is real but it cannot detect which fake product it is. (output3.png and output4.png) -->

<!-- In summary we have 2 main problems:
1.Lack of data
2.Imbalance class
although we had some problems like data Homogeneous probelm that solved and other stuffs data cleaning and preprocessing stuffs but most of them will be in bootcat algorithm.  -->