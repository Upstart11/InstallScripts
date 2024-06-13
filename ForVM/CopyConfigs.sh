#!/bin/bash

#Cloning the dotfiles repo

git clone https://github.com/Upstart11/dotFiles.git

sudo rm -f ~/.nanorc > /dev/null 2>&1
ln -s ~/dotFiles/LinuxCloud/nano/nanorc ~/.nanorc


#creating crontab job

