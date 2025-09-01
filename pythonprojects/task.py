# To-Do List, but make it cute 🌸💖✨

tasks=[]

def show_tasks():
    if not tasks:
        print("🌸 No tasks yet… add something nice! ✨")
    else:
        print("\n💌 Your tasks:")
        for i, task in enumerate(tasks, start=1):
            print(f" {i}. {task} 🌷")

while True:
    print("\nOptions: 1. ➕ Add Task  2. 📖 View Tasks  3. ❌ Remove Task  4. 🌙 Exit")
    choice=int(input("Enter choice (1-4): "))

    if choice==1:
        task=input("🌸 Enter new task: ")
        tasks.append(task)
        print("✨ Task added, yay! 💖")

    elif choice==2:
        show_tasks()

    elif choice==3:
        show_tasks()
        if tasks:
            task_num=int(input("Which one to remove? (number pls) 👉 "))
            if 1<=task_num<=len(tasks):
                removed = tasks.pop(task_num-1)
                print(f"💔 Removed task: {removed} ...but you got this! 💪✨")
                show_tasks()
            else:
                print("⚠️ That’s not a valid number! 🐰")

    elif choice==4:
        print("🌙 Goodbye! Have a magical day! ✨")
        break

    else:
        print("❌ Oops, wrong option, try again pls 💕")
