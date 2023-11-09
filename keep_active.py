import time


def keep_workspace_active():
    # Add your monitoring logic here
    counter = 0
    while True:
        try:
            # Print a message every 2 minutes
            if counter % 120 == 0:
                print("Keeping workspace active...")

            # Increment the counter
            counter += 1

            # Sleep for 1 second
            time.sleep(1)

        except KeyboardInterrupt:
            print("Monitoring interrupted.")
            break


if __name__ == "__main__":
    keep_workspace_active()
