# Project Krasss 
**Members:** Santiago Arciniegas, Sanjeev Dasgupta, Sophie Latham, Trayda Murakami, Konstantina Panagiotopoulou, Alex Richter, Rudranshi Vishnoi

## Research question 
To be updated. 

## Datasets
To be updated. 

## Work flow and development practices
1. Meaningful Commit Messages<br>
`git commit -m "Add user authentication feature"`

2. Branching Strategy<br>
Use branches for new features, bug fixes, and experiments. 
`git checkout -b feature/new-feature`

3. Regularly Pull from Master<br>
Stay up-to-date with the main codebase by regularly pulling changes from the master branch. This reduces merge conflicts and ensures you're working with the latest code.
`git pull origin master`

4. Utilize .gitignore<br>
Keep your repository clean by ignoring files and directories that don't belong in version control, such as logs, build artifacts, and sensitive information.
```
# .gitignore file
logs/
node_modules/
*.env
```

5. Interactive Rebasing<br>
Before merging your code: use interactive rebasing to squash commits, reword messages, and maintain a clean commit history. 
`git rebase -i HEAD~3`

6. Code Reviews and Pull Requests<br>
Encourage a culture of code reviews and pull requests. This collaborative approach ensures code quality, identifies bugs early, and facilitates knowledge sharing.
Pull Request Template:

- Description: Brief overview of changes
- Testing Done: List of tests conducted
- Related Issues: Link to relevant issues

8. Documenting Changes<br>

Be sure to document major changes and be thorough in explanations. Code should be readable and **you should be able to explain all the code you submit!!!!** <br>

Best practices were taken from this [resource](https://dev.to/speaklouder/be-a-better-developer-with-these-git-good-practices-13j9).

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
