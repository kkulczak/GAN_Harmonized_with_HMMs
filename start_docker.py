#!/usr/bin/env python3

import argparse
import getpass
import os
import subprocess
from glob import glob

DEFAULT_NAME = 'GAN_HMM'


def generate_parser():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        '-r', '--runtime',
        choices=['cpu', 'gpu'],
        default='gpu',
        help='runtime version'
    )
    arg_parser.add_argument(
        '-b', '--build',
        action='store_true',
        help='Build docker before running'
    )
    arg_parser.add_argument(
        '-n', '--name',
        default=DEFAULT_NAME,
        help='container name, preceded with "username"'
    )
    arg_parser.add_argument(
        '-m', '--memory-limit',
        default='16g',
        help='Same as docker `-m` flag'
    )
    arg_parser.add_argument(
        '-p', '--project-dir',
        default=os.path.join(os.getcwd(), '..'),
        help='Path to project directory. Mounted in docker under /home/shared'
    )
    arg_parser.add_argument(
        '-d', '--data-dir',
        default='/media/scratch/tsayer/EbagCheckStorage',
        help='Path to  data directory. Mounted in docker under /data'
    )
    arg_parser.add_argument(
        '--build-only',
        action='store_true',
        help='Do not run the container'
    )
    arg_parser.add_argument(
        '--no-ports',
        action='store_true',
        help='disable ports forwarding'
    )
    arg_parser.add_argument(
        '--set-ports',
        nargs='+',
        default=[],
        help='Set ports to docker. First given port will be mapped to docker 6006 port (tensor board), '
             'second - to 8888 (jupyter)',
    )
    arg_parser.add_argument(
        '-c', '--cuda-device',
        type=int,
        help='Set visible cuda devices (Graphic card id when multiple '
             'available) ',
    )
    arg_parser.add_argument(
        '--rm',
        action='store_true'
    )
    arg_parser.add_argument(
        '--cmd',
        default='/bin/bash',
        type=str,
    )
    arg_parser.add_argument(
        '--hidden',
        action='store_true',
        help='No -it flag in run'
    )
    arg_parser.add_argument(
        '--display',
        action='store_true',
        help='Allows displaying graphics',
    )
    arg_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='--no-cache flag during image building'
    )
    return arg_parser


if __name__ == '__main__':
    args = generate_parser().parse_args()


    image_tag = 'gan_hmm'
    container_name = '{}_{}'.format(getpass.getuser(), args.name)

    ports_in_use = ['6006', '8888']

    if args.build or args.build_only:
        optional_build_args = []
        if args.no_cache:
            optional_build_args.append('--no-cache')

        build_args = [
            'docker', 'build',
            '--build-arg', 'UID_NUMBER={}'.format(os.getuid()),
            '--build-arg', 'GID_NUMBER={}'.format(os.getgid()),
            '--build-arg', 'USER_NAME={}'.format(getpass.getuser()),
            '-f', 'Dockerfile',
            '-t', image_tag,
            *optional_build_args,
            '.'
        ]
        print(
            "\n\nStarted docker:\n",
            ' '.join(build_args),
            "\n\n"
        )
        subprocess.run(build_args, check=True)
    if args.build_only:
        exit(0)

    optional_run_args = []
    if args.data_dir:
        optional_run_args += [
            '-v', '{}:/data'.format(args.data_dir)
        ]
    if args.runtime == 'gpu':
        optional_run_args.append('--runtime=nvidia')

    if args.cuda_device is not None:
        optional_run_args += [
            '-e', 'CUDA_VISIBLE_DEVICES={}'.format(args.cuda_device),
            '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID'
        ]
        container_name += '_cd{}'.format(args.cuda_device)

    if not args.no_ports:
        ports = args.set_ports + ports_in_use[len(args.set_ports):]

        for i in range(len(ports_in_use)):
            optional_run_args.extend(
                ['-p', '{}:{}'.format(ports[i], ports_in_use[i])]
            )

    if args.rm:
        optional_run_args.append('--rm')

    if args.hidden:
        optional_run_args.append('-d')
    else:
        optional_run_args.append('-it')
    if args.memory_limit is not None:
        optional_run_args.extend(['-m', args.memory_limit])

    if True:
        optional_run_args.extend(['--cpus=12'])

    if args.display:
        video_paths = glob("/dev/video*")
        optional_run_args.extend([
            '-e', 'DISPLAY={}'.format(os.environ['DISPLAY']),
            '-e', 'QT_X11_NO_MITSHM=1',
            '-e', "TERM=xterm-256color",
            '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
            *[
                '--device={0}:{0}'.format(path)
                for path in video_paths
            ],
        ])

    run_args = [
        'docker', 'run',
        '-v', '{}:/home/shared'.format(args.project_dir),
        '-w', '/home/shared',
        '--name', container_name,
        '--hostname', container_name,
        *optional_run_args,
        image_tag,
        *args.cmd.replace('"', '').split()
    ]

    subprocess.run([
        'docker', 'container', 'rm',
        container_name
    ])
    print(
        "\n\nStarted docker:\n",
        ' '.join(run_args),
        "\n\n"
    )
    subprocess.run(
        run_args
    )
