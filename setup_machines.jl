# SET machine
#   If your path is not here, run homedir() and add if

# Run things on Sherlock (the Stanford-wide cluster)
# Note: The `GROUP_HOME` environmental variable should in theory only exist on Sherlock and not on the Stanford GSB server


if homedir() == "/Users/callende" # Paths Claudia Stanford Desktop



	dbox = "/Users/callende/Dropbox/Research/11. Private Schools Peru/"
	git = "/Users/callende/Desktop/GitHub/JobMarketPaper/"

	const stemWorked = dbox * "3. Data/5. Worked"
	const stemsave = dbox * "4. Work/7. Model/Results"
	const stemGit = git * "Code/ModelJulia"



end

const stemGit = pwd()
print("Paths set for machine: ", split(homedir(), "/")[end])


