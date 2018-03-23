# CapsuleNet solution for Comment toxic classification challenge
[Comment toxic classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) script in kaggle with naive CapsuleNet in MxNet
----
## Some updates for those who stared this project
1. is it working?

    Yes

2. Hows the performance?

    My 10 folds model performed 0.9859 in public board, but only 0.984x in private board.
 
3. Will you update it?

    No. Competition is over. and you'd better not share through Github, since it violents Kaggle Rule.
 
4. How to use it?

    * Install the required Python Libs.
    * python train_k_fold.py for training(multi GPU supported, please refer to MxNet.)
    * python test_k_fold.py for testing(Only single GPU)
  
5. Last thing to mention, I slightly changed the squash function. And I don't like either form of it. 
6. Future work

    * Maybe use it in object detection. Working on MxNet Pikachu example now.
    * Implement the Routing with EM
## Thank you for your Stars.

## Reference
[comment_toxic](https://github.com/jcjview/comment_toxic)
