# GP-Quant
Companion code for the monograph *Gaussian Process Models for Quantitative Finance* by Mike Ludkovski and Jimmy Risk. Last update: November 2024.

### Software and Package Versions

All companion code was ran successfully with the following versions:

* ``R``: 4.2.1 and 4.4.1, ``DiceKriging``: 1.6.0, ``randtoolbox``: 1.31.1, ``rgenoud``: 5.9-0.11, ``ggplot2``: 3.5.1, ``mlOSP``: 1.0 (install from GitHub)
* ``Python``: 3.11.4
  * Packages: ``numpy`` 1.25.2, ``matplotlib`` 3.9.2, ``seaborn`` 0.13.2; **ch2 and ch3**: ``torch`` 2.0.1, ``gpytorch`` 1.11; **ch7**: ``tensorflow`` 2.17, ``tensorflow-probability`` 0.24.0

## Table of Contents

### Chapter 1 Notebook
([Chapter 1 Notebook](./ch1_py.ipynb)) This notebook is in Python and reproduces plots from Chapter 1, demonstrating Gaussian Process surrogates using a synthetic univariate response function $$f(x) = -0.1 x^2 + \sin(3x)$$.  It uses the GPyTorch library.  **IMPORTANT!**  Place `easygpr_helper.py` in the same directory as the notebook for the code to run.  

### Chapter 2 Notebook
([Chapter 2 Notebook](./ch2_py.ipynb)) This notebook is in Python and covers further details of Gaussian Process modeling, comparing different kernel configurations and hyperparameter tuning across varying training set sizes. It also uses GPyTorch and requires the `easygpr_helper.py` file for execution. 

### Chapter 4 Notebook
([Chapter 4 Notebook](./ch4_R.pdf)) ([.Rmd](./ch4_R.Rmd)) This notebook is in R and illustrates the use of Gaussian Process surrogates for estimating option prices and Greeks, focusing on both the Heston and Black-Scholes models. It demonstrates GP fitting, hyperparameter optimization, and inference of Delta, Theta, and Gamma using both Squared-Exponential and Matern-5/2 kernels. The notebook relies on the ``DiceKriging`` GP library and the ``randtoolbox`` library for QMC sampling.

### Chapter 5 Notebook
([Chapter 5 Notebook](./ch5_R.pdf)) ([.Rmd](./ch5_R.Rmd)) This notebook is in R and tackles Bermudan option pricing with Gaussian Processes using the Regression Monte Carlo (RMC) approach. It applies the [``mlOSP``](https://github.com/mludkov/mlOSP) package to build emulators for Bermudan Put options, showing both one- and two-dimensional examples. It includes dynamic emulation, sequential design, and adaptive batching techniques.

### Chapter 6 Notebook
([Chapter 6 Notebook](./ch6_py.ipynb)) This notebook uses Python and TensorFlow to construct a Gaussian Process model for natural gas forward curves, modeling annual seasonality with additive kernels. It uses the TensorFlow library and reproduces Figure 6.2 from Section 6.2. 


## License

All code is copyrighted under the MIT License; see [link](./LICENSE) for details.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
