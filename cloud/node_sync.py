import os
import time
import moxing as mox


def remove_remote_sync_files(remote_path, rank):
    s3_rank_id_fn = os.path.join(remote_path, 'rank_{}.txt'.format(rank))
    if mox.file.exists(s3_rank_id_fn):
        mox.file.remove(s3_rank_id_fn, recursive=False)


def sync_through_remote_file(remote_path, rank, world_size):
    s3_rank_id_fn = os.path.join(remote_path, 'rank_{}.txt'.format(rank))

    # create a file meaning that the job in this process is done
    mox.file.write(s3_rank_id_fn, '{}'.format(rank))

    # multi node, wait for other rank transfer data
    while True:
        all_rank_exist = True
        for rank_item in range(world_size):
            rank_fn_item = os.path.join(remote_path, 'rank_{}.txt'.format(rank_item))
            if not mox.file.exists(rank_fn_item):
                all_rank_exist = False
        if all_rank_exist:
            break
        else:
            time.sleep(5) # delay 5 sec
