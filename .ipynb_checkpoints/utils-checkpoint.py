{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'X' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-4-51011713172f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mX2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mY2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m \u001b[0mX2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrandomize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mdraw_learning_curves\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mestimator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_trainings\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'X' is not defined"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import learning_curve\n",
    "\n",
    "##should be read together with detect_overfiting.py file from where this code is called\n",
    "\n",
    "# It is good to randomize the data before drawing Learning Curves\n",
    "def randomize(X, Y):\n",
    "    permutation = np.random.permutation(Y.shape[0])\n",
    "    X2 = X[permutation,:]\n",
    "    Y2 = Y[permutation]\n",
    "    return X2, Y2\n",
    "\n",
    "X2, y2 = randomize(X, y)\n",
    "\n",
    "def draw_learning_curves(X, y, estimator, num_trainings):\n",
    "    train_sizes, train_scores, test_scores = learning_curve(\n",
    "        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))\n",
    "\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    train_scores_std = np.std(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    test_scores_std = np.std(test_scores, axis=1)\n",
    "\n",
    "    plt.grid()\n",
    "\n",
    "    plt.title(\"Learning Curves\")\n",
    "    plt.xlabel(\"Training examples\")\n",
    "    plt.ylabel(\"Score\")\n",
    "\n",
    "    plt.plot(train_scores_mean, 'o-', color=\"g\",\n",
    "             label=\"Training score\")\n",
    "    plt.plot(test_scores_mean, 'o-', color=\"y\",\n",
    "             label=\"Cross-validation score\")\n",
    "\n",
    "\n",
    "    plt.legend(loc=\"best\")\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
