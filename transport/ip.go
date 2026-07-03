package transport

import "net"

func isPrivateOrMetadataIP(ip net.IP) bool {
	if ip == nil {
		return false
	}
	return ip.IsLoopback() ||
		ip.IsUnspecified() ||
		ip.Equal(net.IPv4bcast) ||
		ip.IsPrivate() ||
		ip.IsMulticast() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		isCarrierGradeNATIP(ip) ||
		ip.Equal(net.ParseIP("169.254.169.254")) ||
		ip.Equal(net.ParseIP("169.254.170.2")) ||
		ip.Equal(net.ParseIP("fd00:ec2::254"))
}

func isCarrierGradeNATIP(ip net.IP) bool {
	ip4 := ip.To4()
	if ip4 == nil {
		return false
	}
	return ip4[0] == 100 && ip4[1]&0xc0 == 0x40
}
