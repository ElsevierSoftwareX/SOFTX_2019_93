import os
import sys

# We will use the linux command sed
command = "sed -i "

# Mathematica can represent certain expressions in multiple ways. These
# first replacements standardise these representations. In general
# convert things to subscript boxes
stand = {\
    r"\\\*SubsuperscriptBox\[\\(\(\\\[[]\[A-Za-z]*]\)\\), \\(\([0-9]*\)\\), \\(\([0-9]*\)\\)]":
        r"Subscript\[\1, \2]^\3",\
    r"\\\*SuperscriptBox\[\\(\(\\\[[]\[A-Za-z]*]\)\\), \\(\([0-9]*\)\\)]":\
        r"(\1^(\2))"}
standC = command
for key, value in stand.items():
    standC += " -e 's:%s:%s:g'"%(key,value)
standC += " %s"%sys.argv[2]

# We now make subsitutions for the variables we are using
var = {\
    r"Subscript\[\\\[Alpha], 2]":r"a2",\
    r"Subscript\[\\\[Alpha], 0]":r"a0",\
    r"C\[1]":r"C1",\
    r"C\[2]":r"C2",\
    r"Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 0]\[0,1]":r"phi0[-1]",\
    r"Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 1]\[0,1]":r"phi1[-1]",\
    r"Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 2]\[0,1]":r"phi2[-1]",\
    r"Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 3]\[0,1]":r"phi3[-1]",\
    r"Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 4]\[0,1]":r"phi4[-1]",\
    r"\\\[Kappa]\[\\\[Rho]]":r"kappa",\
    r"\\\[Tau]":r"t",\
    r"\\\[Rho]":r"r",\
    r"\\\[Mu]\[1]":r"mu[-1]",\
    r"(\\\[Mu]^\\\[Prime])\[1]":r"mup[-1]",\
    r"\\\[Kappa]\[1]":r"kappa[-1]",\
    r"(\\\[Kappa]^\\\[Prime])\[1]":r"kappap[-1]",\
    r"((Subscript\[\\\[CapitalTheta]\\\[CapitalPhi], 0])^(0,1))\[0,1]":\
        r"self.exactValue(0,1,sDer =1).fields[0][-1]"}
varC = command
for key, value in var.items():
    varC += " -e 's:%s:%s:g'"%(key,value)
varC += " %s"%sys.argv[2]

# Now we make subsitutions for the compound statements
comp = {\
    r"Sqrt\[9+4 a2^2]":r"C",\
    r"r/(r\^2-t\^2 kappa\^2)":r"rR",\
    r"r-t kappa":r"Rm",\
    r"r+t kappa":r"Rp"}
    
compC = command
for key, value in comp.items():
    compC += " -e 's:%s:%s:g'"%(key,value)
compC += " %s"%sys.argv[2]

# Remove left overs
delt = {\
    r"\\!":r""}
deltC = command
for key, value in delt.items():
    deltC += " -e 's:%s:%s:g'"%(key,value)
deltC += " %s"%sys.argv[2]

# Replace Mathematica operators with python operators
ope = {r" ":r"*",\
    r"\^":r"**"}
opeC= command
for key, value in ope.items():
    opeC += " -e 's:%s:%s:g'"%(key,value)
opeC += " %s"%sys.argv[2]

# First we strip all of the end of line characters
os.system(r"sed ':a;N;$!ba;s:\n::g' <%s >%s"%(sys.argv[1],sys.argv[2]))

# Now we perform the subsitutions
os.system(deltC)
os.system(standC)
os.system(varC)
os.system(compC)
os.system(opeC)
