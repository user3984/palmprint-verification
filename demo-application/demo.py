from auth_service import Service

service = Service()
# service = Service("store/yc.pkl")

while True:
    cmd = input("cmd: ").strip()
    if cmd == "roi":
        service.demo_roi_camera()
    elif cmd == "detect":
        verify_id = service.verify_id()
        if verify_id is None:
            print("Palmprint not recognized.")
        else:
            print("Palmprint recognized, user: ", verify_id)
    elif cmd == "add":
        name = input("Input user name: ")
        if service.user_exists(name):
            print(f"User {name} exists.")
            continue
        else:
            service.add_user(name)
    elif cmd == "save":
        fname = input("Input filename: ")
        service.save_db(fname)
    elif cmd == "lsusers":
        for u in service.users:
            print(f"    {u}")
        print()
    elif cmd == "exit":
        break
