import os
import json
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent
import uuid

base_dir = '/tmp/lmtracer_exports'

filenames = os.listdir(base_dir)
pids = set()

for filename in filenames:
    if filename.startswith('trace_data_') and filename.endswith('.bin'):
        filename_split = filename.split('_')
        hostname = filename_split[2]
        hostname_pid = filename_split[3].split('.')[0]
        pids.add(f'{hostname}_{hostname_pid}')
    
trace_builder = TraceProtoBuilder()
trace_points = {}
pid_to_time_offset = {}

for hostname_pid in pids:
    trace_points_json = json.load(open(os.path.join(base_dir, f'metadata_{hostname_pid}.json'), 'r'))
    for key in trace_points_json:
        trace_points[key] = trace_points_json[key]

for hostname_pid in pids:
    trace_point_id_to_name = {}
    for key in trace_points:
        trace_point_id_to_name[trace_points[key]] = key
    
    packet = trace_builder.add_packet()
    packet.track_descriptor.name = hostname_pid
    packet.track_descriptor.uuid = uuid.uuid4().int & ((1 << 63) - 1)
    current_track_uuid = packet.track_descriptor.uuid
    
    op_name_begin = {}

    filename = os.path.join(base_dir, f'trace_data_{hostname_pid}.bin')

    with open(filename, 'rb') as f:
        data = f.read()
        num_clock_values = len(data) // 16
        for i in range(num_clock_values):
            trace_point_id = int.from_bytes(data[i*16:i*16+8], byteorder='little')
            # original = int(int.from_bytes(data[i*16+8:(i+1)*16], byteorder='little'))
            # print(f'{original} - {pid_to_time_offset[pid]} = {original - pid_to_time_offset[pid]}')
            # timestamp = int(int.from_bytes(data[i*16+8:(i+1)*16], byteorder='little')) + pid_to_time_offset[pid]
            timestamp = int.from_bytes(data[i*16+8:(i+1)*16], byteorder='little')
            op_name_start_end = trace_point_id_to_name.get(trace_point_id, f'unknown_op_{trace_point_id}')
            op_name = op_name_start_end.split('_')[-2]
            start_or_end = op_name_start_end.split('_')[-1]

            if start_or_end == 'start':
                if op_name in op_name_begin:
                    print(f'Warning: duplicate start for op {op_name}')
                    # packet = trace_builder.add_packet()
                    # packet.timestamp = op_name_begin[op_name]
                    # packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                    # packet.track_event.name = op_name
                    # packet.track_event.track_uuid = current_track_uuid

                    # packet.trusted_packet_sequence_id = 1

                    # packet = trace_builder.add_packet()
                    # packet.timestamp = timestamp
                    # packet.track_event.type = TrackEvent.TYPE_SLICE_END
                    # packet.track_event.name = op_name
                    # packet.track_event.track_uuid = current_track_uuid

                    # packet.trusted_packet_sequence_id = 1
                op_name_begin[op_name] = timestamp
            elif start_or_end == 'end':
                if op_name in op_name_begin:
                    packet = trace_builder.add_packet()
                    packet.timestamp = op_name_begin[op_name]
                    packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                    packet.track_event.name = op_name
                    packet.track_event.track_uuid = current_track_uuid

                    packet.trusted_packet_sequence_id = 1

                    packet = trace_builder.add_packet()
                    packet.timestamp = timestamp
                    packet.track_event.type = TrackEvent.TYPE_SLICE_END
                    packet.track_event.name = op_name
                    packet.track_event.track_uuid = current_track_uuid

                    packet.trusted_packet_sequence_id = 1
                    
                    del op_name_begin[op_name]

trace_data = trace_builder.serialize()
with open('./lmtracer_trace_output.perfetto', 'wb') as f:
    f.write(trace_data)

