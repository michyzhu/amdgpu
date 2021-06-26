# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs
export HOME=/home/myz
PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH=/home/myz/miniconda3/bin:$PATH
export SINGULARITY_DISABLE_CACHE=true


