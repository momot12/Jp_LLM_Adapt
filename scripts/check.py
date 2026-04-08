p="一文のみでの回答。"

# sys_prompt="日本語で一文のみでの回答。"

other = "自分は男性。一文のみでの回答。"

print(other.rstrip('。').split('。'))
print(len(other.rstrip('。').split('。')))

print(p.rstrip('。').split('。'))
print(len(p.rstrip('。').split('。')))

if len(other.rstrip('。').split('。')) > 1:
    b = other.rstrip('。').split('。')
    a = b[0]+'。日本語で'+b[1]+'。'
    
    print(a)
    