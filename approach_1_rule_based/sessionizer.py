import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to sys.path so we can import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.models import Turn, Conversation, merge_into_turns
from shared.data_loader import prepare_messages

def sessionize_personal_chats(
    chats: List[Dict],
    idle_gap_seconds: int = 10800,
    burst_window_seconds: int = 90,
    min_turns: int = 2,
    require_target_user: bool = True,
) -> List[Conversation]:
    conversations = []
    conv_id_counter = 1
    
    for chat in chats:
        messages = prepare_messages(chat)
        if not messages:
            continue
            
        # Keep track of which messages are forwarded
        is_forwarded_msg = { m.get("id"): ("forwarded_from" in m) for m in messages if "id" in m }
            
        turns = merge_into_turns(messages, burst_window_seconds=burst_window_seconds)
        if not turns:
            continue
            
        # 1. Idle-Gap Session Split
        raw_sessions = []
        current_session = [turns[0]]
        for turn in turns[1:]:
            if turn.start_unixtime - current_session[-1].end_unixtime > idle_gap_seconds:
                raw_sessions.append(current_session)
                current_session = [turn]
            else:
                current_session.append(turn)
        if current_session:
            raw_sessions.append(current_session)
            
        # 2. Reply Chain Attachment
        parent = {i: i for i in range(len(raw_sessions))}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if root_i < root_j:
                    parent[root_j] = root_i
                else:
                    parent[root_i] = root_j
                    
        # Map message_id to session index
        msg_id_to_session_idx = {}
        for idx, session in enumerate(raw_sessions):
            for turn in session:
                for msg_id in turn.message_ids:
                    msg_id_to_session_idx[msg_id] = idx
                    
        for idx, session in enumerate(raw_sessions):
            for turn in session:
                reply_id = turn.reply_to_message_id
                if reply_id is not None and reply_id in msg_id_to_session_idx:
                    target_idx = msg_id_to_session_idx[reply_id]
                    root_current = find(idx)
                    root_target = find(target_idx)
                    if root_current != root_target:
                        early_idx = min(idx, target_idx)
                        late_idx = max(idx, target_idx)
                        gap = raw_sessions[late_idx][0].start_unixtime - raw_sessions[early_idx][-1].end_unixtime
                        # If ended less than 12 hours ago (12 * 3600 = 43200 seconds)
                        if gap < 43200:
                            union(idx, target_idx)
                            
        from collections import defaultdict
        merged_groups = defaultdict(list)
        for i in range(len(raw_sessions)):
            merged_groups[find(i)].append(i)
            
        final_sessions = []
        for root in sorted(merged_groups.keys()):
            session_turns = []
            for i in merged_groups[root]:
                session_turns.extend(raw_sessions[i])
            session_turns.sort(key=lambda t: t.start_unixtime)
            final_sessions.append(session_turns)
            
        # 3. Training Quality Filter & Object Creation
        for session_turns in final_sessions:
            if len(session_turns) < min_turns:
                continue
                
            target_user_turns = sum(1 for t in session_turns if t.is_target_user)
            if require_target_user and target_user_turns == 0:
                continue
                
            # Check if all messages are forwarded content only
            all_msg_ids = [msg_id for turn in session_turns for msg_id in turn.message_ids]
            if all_msg_ids and all(is_forwarded_msg.get(msg_id, False) for msg_id in all_msg_ids):
                continue
                
            conv = Conversation(
                conversation_id=f"{chat.get('id', 0)}_{conv_id_counter}",
                chat_id=chat.get("id", 0),
                chat_name=chat.get("name", "Unknown"),
                chat_type=chat.get("type", "personal_chat"),
                approach="rule_based",
                turns=session_turns
            )
            conv.finalize()
            conversations.append(conv)
            conv_id_counter += 1
            
    return conversations
