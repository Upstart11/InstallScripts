#!/bin/bash

#Cloning the dotfiles repo

git clone https://github.com/Upstart11/dotFiles.git

# nano
sudo rm -f ~/.nanorc > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/nano/nanorc ~/.nanorc

# bash
sudo rm -f ~/.bashrc > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/bash/.bashrc ~/.bashrc

# tmux
sudo rm -f ~/.tmux.conf > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/tmux/.tmux.conf ~/.tmux.conf

# btop
sudo rm -f ~/.config/btop/btop.conf > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/.config/btop/btop.conf ~/.config/btop/btop.conf

# ranger
sudo rm -f ~/.config/ranger/rifle.conf > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/.config/ranger/rifle.conf ~/.config/ranger/rifle.conf

sudo rm -f ~/.config/ranger/rc.conf > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/.config/ranger/rc.conf ~/.config/ranger/rc.conf

#creating crontab job

