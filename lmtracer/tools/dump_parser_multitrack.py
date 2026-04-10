import os
import json
import uuid

from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

base_dir = '/tmp/lmtracer_exports'
SEQ_ID = 1

trace_builder = TraceProtoBuilder()

filenames = os.listdir(base_dir)
pids = set()

for filename in filenames:
    if filename.startswith('trace_data_') and filename.endswith('.bin'):
        filename_split = filename.split('_')
        hostname = filename_split[2]
        hostname_pid = filename_split[3].split('.')[0]
        pids.add(f'{hostname}_{hostname_pid}')


trace_points = {}

for hostname_pid in pids:
    trace_points_json = json.load(open(os.path.join(base_dir, f'metadata_{hostname_pid}.json'), 'r'))
    for key in trace_points_json:
        trace_points[key] = trace_points_json[key]


for pid_str in pids:

    print(f'Processing pid: {pid_str}')

    op_name_begin = {}

    hostname, pid_num = pid_str.rsplit('_', 1)
    pid_num = int(pid_num)

    packet = trace_builder.add_packet()
    packet.process_descriptor.pid = pid_num
    packet.process_descriptor.process_name = (
        f'{hostname}:{pid_num}'
    )

    process_track_uuid = uuid.uuid4().int & ((1 << 63) - 1)

    packet = trace_builder.add_packet()
    packet.track_descriptor.uuid = process_track_uuid
    packet.track_descriptor.name = f'process:{pid_num}'
    packet.track_descriptor.process.pid = pid_num

    ops_group_uuid = uuid.uuid4().int & ((1 << 63) - 1)

    packet = trace_builder.add_packet()
    packet.track_descriptor.uuid = ops_group_uuid
    packet.track_descriptor.name = 'ops'
    packet.track_descriptor.parent_uuid = process_track_uuid

    op_to_track_uuid = {}

    def get_op_track(op_name: str) -> int:
        if op_name not in op_to_track_uuid:
            packet = trace_builder.add_packet()
            packet.track_descriptor.uuid = (
                uuid.uuid4().int & ((1 << 63) - 1)
            )
            packet.track_descriptor.name = op_name
            packet.track_descriptor.parent_uuid = ops_group_uuid
            op_to_track_uuid[op_name] = packet.track_descriptor.uuid
        return op_to_track_uuid[op_name]

    trace_point_id_to_name = {
        v: k for k, v in trace_points.items()
    }

    filename = os.path.join(
        base_dir, f'trace_data_{pid_str}.bin'
    )

    with open(filename, 'rb') as f:
        data = f.read()
        num = len(data) // 16

        for i in range(num):
            trace_point_id = int.from_bytes(
                data[i*16 : i*16 + 8], 'little'
            )
            timestamp = int.from_bytes(
                data[i*16 + 8 : (i + 1)*16], 'little'
            )

            name = trace_point_id_to_name.get(
                trace_point_id,
                f'unknown_{trace_point_id}'
            )

            parts = name.split('_')
            if len(parts) < 2:
                continue

            op_name = parts[-2]
            start_or_end = parts[-1]

            track_uuid = get_op_track(op_name)

            
            if start_or_end == 'start':
                if op_name in op_name_begin:
                    print(f'Warning: duplicate start for op {op_name}')
                op_name_begin[op_name] = timestamp
            elif start_or_end == 'end':
                if op_name in op_name_begin:
                    packet = trace_builder.add_packet()
                    packet.timestamp = op_name_begin[op_name]
                    packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                    packet.track_event.name = op_name
                    packet.track_event.track_uuid = track_uuid

                    packet.trusted_packet_sequence_id = 1

                    packet = trace_builder.add_packet()
                    packet.timestamp = timestamp
                    packet.track_event.type = TrackEvent.TYPE_SLICE_END
                    packet.track_event.name = op_name
                    packet.track_event.track_uuid = track_uuid

                    packet.trusted_packet_sequence_id = 1
                    
                    del op_name_begin[op_name]

trace_data = trace_builder.serialize()
with open('lmtracer_trace_output.perfetto', 'wb') as f:
    f.write(trace_data)