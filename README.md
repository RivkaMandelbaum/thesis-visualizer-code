# thesis-visualizer-code
<h2>Files</h2>

<ul><li>demo-data/ contains demo data for the visualizer </li>

<li>experiment-code, experiment-data, and experiment-data-analysis contain code, data, and analysis files from the experiment described in Section 2. </li>

<li>The remaining files are for running the visualizer itself. Instructions for running the visualizer can be found below. </li></ul>

<h2>Requirements</h2>
This project was developed with PsyNet 9.4.1 (using Dallinger 9.3.1). The PsyNet installation instructions can be found here: https://psynetdev.gitlab.io/PsyNet/developer_installation/index.html. 
The visualizer works with networkx 2.6.3 and PyVis 0.2.1. 

<h2>Running the Visualizer</h2>

To run the visualizer, you can follow these steps: 
<ol>
<li>Clone the visualizer</li>
<li>If using a PsyNet virtual environment, make the visualizer a subdirectory of PsyNet and activate your virtual environment </li>
<li>cd into the visualizer directory</li>
<li>To run with demo data, use the command python3 runserver.py [port] -t. The visualizer should start running locally with the demo data. </li>
<ul><li>To run with your own data, use the command python3 runserver.py [port] [directory name]. The data must be in the PsyNet export format.</li></ul>
</ol>
